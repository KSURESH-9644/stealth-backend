import 'dotenv/config';
import { app, BrowserWindow, screen, ipcMain, desktopCapturer, clipboard, dialog } from 'electron';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import screenshot from 'screenshot-desktop';
import { uIOhook, UiohookKey } from 'uiohook-napi';
import { GoogleGenAI } from '@google/genai';
import Groq, { toFile } from 'groq-sdk';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Dynamic Resume Context Loading (Supports JSON, TXT, MD, etc.)
let resumeProfileText = '';
function loadResumeFile(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      try {
        const parsed = JSON.parse(raw);
        resumeProfileText = JSON.stringify(parsed, null, 2);
      } catch {
        resumeProfileText = raw; // Fallback for plain text / markdown
      }
      console.log(`✅ Resume loaded: ${filePath}`);
      broadcast({ type: 'status', msg: '✅ Resume Context Synced' });
      return true;
    }
  } catch (err) {
    console.error('Failed to load resume file:', err);
  }
  return false;
}

// Default load check
loadResumeFile(path.join(__dirname, 'resumeContext.json'));

const rawKeys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const geminiKeys = rawKeys.split(',').map(k => k.trim()).filter(Boolean);
let currentKeyIndex = 0;

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

let overlayWindow = null;
let imageBuffer = [];
let clients = [];
let currentMode = 'speech';
let lastAnswerText = '';

const expressApp = express();
expressApp.use(express.json({ limit: '50mb' }));
const server = createServer(expressApp);
const wss = new WebSocketServer({ server });

expressApp.use(express.static(path.join(__dirname, 'public')));

wss.on('connection', (ws) => {
  clients.push(ws);
  ws.on('close', () => { clients = clients.filter(c => c !== ws); });
});

function broadcast(payload) {
  clients.forEach(c => { if (c.readyState === 1) c.send(JSON.stringify(payload)); });
  if (overlayWindow && !overlayWindow.isDestroyed()) {
    overlayWindow.webContents.send('ai-update', payload);
  }
}

// Smart AI Routing with Active Models
async function executeWithFallback({ prompt, images = [], isVision = false }) {
  // 1️⃣ SCENARIO A: VISION / SCREENSHOTS -> GEMINI FIRST
  if (isVision && images.length > 0) {
    let attempts = 0;
    const maxAttempts = geminiKeys.length;

    while (attempts < maxAttempts) {
      const key = geminiKeys[currentKeyIndex];
      try {
        const ai = new GoogleGenAI({ apiKey: key });
        const contents = [prompt, ...images];
        const response = await ai.models.generateContent({
          model: 'gemini-3.6-flash',
          contents: contents
        });
        return response.text;
      } catch (err) {
        console.warn(`⚠️ Gemini Vision Key [${currentKeyIndex + 1}/${maxAttempts}] failed:`, err.message || err);
        currentKeyIndex = (currentKeyIndex + 1) % geminiKeys.length;
        attempts++;
      }
    }

    console.log('🔄 Gemini Vision failed -> Falling back to Groq Vision...');
    try {
      const groqImageContent = images.map(img => ({
        type: 'image_url',
        image_url: { url: `data:${img.inlineData.mimeType};base64,${img.inlineData.data}` }
      }));

      const chatCompletion = await groq.chat.completions.create({
        model: 'llama-3.2-11b-vision-preview',
        messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, ...groqImageContent] }]
      });
      return chatCompletion.choices[0]?.message?.content || '';
    } catch (groqErr) {
      console.error('❌ Groq Vision backup failed:', groqErr.message || groqErr);
      throw new Error(`Vision pipeline failed: ${groqErr.message || groqErr}`);
    }
  }

  // 2️⃣ SCENARIO B: VOICE / AUDIO QUESTIONS -> GROQ FIRST
  else {
    try {
      const chatCompletion = await groq.chat.completions.create({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: prompt }]
      });
      return chatCompletion.choices[0]?.message?.content || '';
    } catch (groqErr) {
      console.warn('⚠️ Groq Voice LLM failed -> Falling back to Gemini pool:', groqErr.message || groqErr);

      let attempts = 0;
      const maxAttempts = geminiKeys.length;
      while (attempts < maxAttempts) {
        const key = geminiKeys[currentKeyIndex];
        try {
          const ai = new GoogleGenAI({ apiKey: key });
          const response = await ai.models.generateContent({
            model: 'gemini-3.6-flash',
            contents: [prompt]
          });
          return response.text;
        } catch (err) {
          console.warn(`Gemini Backup Key [${currentKeyIndex + 1}/${maxAttempts}] failed:`, err.message || err);
          currentKeyIndex = (currentKeyIndex + 1) % geminiKeys.length;
          attempts++;
        }
      }
      throw new Error(`Voice pipeline failed: ${groqErr.message || groqErr}`);
    }
  }
}

function getModePrompt() {
  const profileSnippet = resumeProfileText
    ? `\n[CANDIDATE PROFILE & STACK]:\n${resumeProfileText}\n`
    : '';

  if (currentMode === 'code') {
    return (
      "Act as a pragmatic senior full-stack engineer in a live coding interview." +
      profileSnippet +
      "Rules for Code Output:\n" +
      "1. Return ONLY clean, optimal, production-ready code.\n" +
      "2. Minimal inline comments explaining 'why'.\n" +
      "3. Bottom line: Time and Space Complexity in 1 line.\n" +
      "4. ZERO conversational text, greetings, or explanations before/after code."
    );
  } else {
    return (
      "Act as a direct speaking assistant for a senior full-stack engineer (.NET Core & Vue 3) in a live interview." +
      profileSnippet +
      "STRICT BREVITY & SPEAKING RULES:\n" +
      "1. Max 3 to 4 short, crisp bullet points total (maximum 50-70 words overall).\n" +
      "2. Natural speech format: Write exactly what the candidate should say verbally in 20 seconds.\n" +
      "3. NO filler phrases (DO NOT write 'That is an interesting question', 'Sure', 'From my perspective').\n" +
      "4. Focus strictly on real engineering concepts, runtime behavior, and practical usage.\n" +
      "5. If context applies, align with Vue 3 / Pinia or .NET Core / SQL Server from candidate profile."
    );
  }
}

function createOverlayWindow() {
  const { width } = screen.getPrimaryDisplay().workAreaSize;

  overlayWindow = new BrowserWindow({
    width: 520,
    height: 440,
    x: width - 540,
    y: 30,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    focusable: true,
    backgroundColor: '#00000000',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  overlayWindow.setContentProtection(true);
  overlayWindow.loadFile(path.join(__dirname, 'overlay.html'));
}

ipcMain.handle('get-desktop-source-id', async () => {
  const sources = await desktopCapturer.getSources({ types: ['screen'] });
  return sources[0]?.id;
});

// Enhanced file selector allowing JSON, Text, Markdown, Docs
ipcMain.handle('select-resume-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    title: 'Select Resume / Profile File',
    filters: [
      { name: 'Supported Files', extensions: ['json', 'txt', 'md', 'pdf', 'docx'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    properties: ['openFile']
  });

  if (!canceled && filePaths.length > 0) {
    const success = loadResumeFile(filePaths[0]);
    return { success, path: filePaths[0], name: path.basename(filePaths[0]) };
  }
  return { success: false };
});

ipcMain.on('window-minimize', () => overlayWindow?.minimize());
ipcMain.on('window-toggle-maximize', () => {
  if (!overlayWindow) return;
  overlayWindow.isMaximized() ? overlayWindow.unmaximize() : overlayWindow.maximize();
});
ipcMain.on('window-hide', () => overlayWindow?.hide());

async function captureScreen() {
  try {
    const imgBuffer = await screenshot({ format: 'png' });
    imageBuffer.push({ inlineData: { data: imgBuffer.toString('base64'), mimeType: 'image/png' } });
    broadcast({ type: 'status', msg: `Captured: ${imageBuffer.length} | Mode: ${currentMode.toUpperCase()}` });
  } catch (err) { console.error(err); }
}

async function solveVisionQuestion() {
  if (imageBuffer.length === 0) {
    broadcast({ type: 'status', msg: 'No screenshots captured!' });
    return;
  }
  broadcast({ type: 'status', msg: `Analyzing [${currentMode.toUpperCase()}]...` });
  try {
    const prompt = getModePrompt() + "\nAnalyze this technical problem on screen and solve it directly:";
    const answer = await executeWithFallback({ prompt, images: imageBuffer, isVision: true });
    lastAnswerText = answer;
    broadcast({ type: 'answer', text: answer });
  } catch (err) {
    broadcast({ type: 'status', msg: 'Vision Error: ' + err.message });
  } finally {
    imageBuffer = [];
  }
}

ipcMain.on('process-internal-audio', async (event, base64Audio) => {
  broadcast({ type: 'status', msg: 'Transcribing speech...' });
  try {
    const audioBuffer = Buffer.from(base64Audio, 'base64');
    const audioFile = await toFile(audioBuffer, 'audio.webm', { type: 'audio/webm' });
    const transcription = await groq.audio.transcriptions.create({
      file: audioFile,
      model: 'whisper-large-v3-turbo',
      language: 'en'
    });

    const questionText = transcription.text;
    if (!questionText || !questionText.trim() || questionText.length < 4) {
      broadcast({ type: 'status', msg: 'No clear speech detected' });
      return;
    }

    broadcast({ type: 'status', msg: `Heard: "${questionText.slice(0, 25)}..." -> Thinking...` });
    const prompt = getModePrompt() + `\n\nInterviewer Question: "${questionText}"\nProvide the direct response:`;
    const answer = await executeWithFallback({ prompt, isVision: false });
    lastAnswerText = answer;
    broadcast({ type: 'answer', text: `🎙️ "${questionText}"\n\n💡 ${answer}` });
  } catch (err) {
    broadcast({ type: 'status', msg: 'Audio Error: ' + err.message });
  }
});

uIOhook.on('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey) {
    if (e.keycode === UiohookKey.C) captureScreen();
    else if (e.keycode === UiohookKey.S) solveVisionQuestion();
    else if (e.keycode === UiohookKey.V) overlayWindow?.webContents.send('toggle-voice');
    else if (e.keycode === UiohookKey.D) overlayWindow?.webContents.send('toggle-direct-mode');
    else if (e.keycode === UiohookKey.X) { imageBuffer = []; broadcast({ type: 'clear' }); }
    else if (e.keycode === UiohookKey.H) overlayWindow?.isVisible() ? overlayWindow.hide() : overlayWindow.show();
    else if (e.keycode === UiohookKey.N1 || e.keycode === UiohookKey.Num1) { currentMode = 'code'; broadcast({ type: 'status', msg: 'Mode: CODING ONLY' }); }
    else if (e.keycode === UiohookKey.N2 || e.keycode === UiohookKey.Num2) { currentMode = 'speech'; broadcast({ type: 'status', msg: 'Mode: SPEECH / BULLETS' }); }
    else if (e.keycode === UiohookKey.Q && lastAnswerText) { clipboard.writeText(lastAnswerText); broadcast({ type: 'status', msg: '📋 Copied to Clipboard!' }); }
  }
});

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (overlayWindow) {
      if (overlayWindow.isMinimized()) overlayWindow.restore();
      overlayWindow.show();
      overlayWindow.focus();
    }
  });

  app.whenReady().then(() => {
    createOverlayWindow();
    uIOhook.start();
    server.listen(3000, '0.0.0.0', () => console.log('Stealth Assistant running on port 3000'));
  });
}

app.on('before-quit', () => server.close());
app.on('window-all-closed', () => { uIOhook.stop(); if (process.platform !== 'darwin') app.quit(); });