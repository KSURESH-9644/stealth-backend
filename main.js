import 'dotenv/config';

import {
  app,
  BrowserWindow,
  screen,
  ipcMain,
  desktopCapturer,
  clipboard,
  dialog
} from 'electron';

import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import screenshot from 'screenshot-desktop';
import { uIOhook, UiohookKey } from 'uiohook-napi';
import { GoogleGenAI } from '@google/genai';
import Groq, { toFile } from 'groq-sdk';
import sharp from 'sharp';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { performance } from 'perf_hooks';

/* ============================================================
   PATH & ENVIRONMENT
============================================================ */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ============================================================
   GLOBAL STATE
============================================================ */
let overlayWindow = null;
let imageBuffer = [];
let clients = [];
let currentMode = 'speech';
let lastAnswerText = '';
let resumeProfileText = '';
let currentGeminiKeyIndex = 0;
let isVisionBusy = false;
let isAudioBusy = false;
let directVoiceEnabled = false;
let isGeminiEnabled = true;

/* ============================================================
   API KEYS & CLIENTS
============================================================ */
const rawGeminiKeys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const geminiKeys = rawGeminiKeys
  .split(',')
  .map(k => k.trim())
  .filter(k => k.length > 0);

const groqApiKey = process.env.GROQ_API_KEY || '';
const groq = groqApiKey ? new Groq({ apiKey: groqApiKey }) : null;

/* ============================================================
   PRODUCTION MODEL POOLS
============================================================ */
const GEMINI_MODELS = [
  'gemini-3.8-flash',
  'gemini-3.7-flash',
  'gemini-3.6-flash',
  'gemini-3.5-flash'
];

const GROQ_VOICE_MODELS = [
  'openai/gpt-oss-20b',
  'openai/gpt-oss-120b'
];

const GROQ_VISION_MODELS = [
  'qwen/qwen3.8-27b',
  'qwen/qwen3.6-27b'
];

const WHISPER_MODEL = 'whisper-large-v3-turbo';
const TEXT_MAX_TOKENS = 600;
const VISION_MAX_TOKENS = 600;

/* ============================================================
   EXPRESS & WEBSOCKET
============================================================ */
const expressApp = express();
expressApp.use(express.json({ limit: '50mb' }));
expressApp.use(express.static(path.join(__dirname, 'public')));

const server = createServer(expressApp);
const wss = new WebSocketServer({ server });

wss.on('connection', ws => {
  clients.push(ws);
  console.log(`WebSocket client connected. Total: ${clients.length}`);

  try {
    ws.send(JSON.stringify({
      type: 'direct-voice-state',
      enabled: directVoiceEnabled
    }));
  } catch {}

  ws.on('close', () => {
    clients = clients.filter(c => c !== ws);
  });
});

/* ============================================================
   BROADCAST
============================================================ */
function broadcast(payload) {
  if (Array.isArray(clients)) {
    clients.forEach(client => {
      try {
        if (client && client.readyState === 1) {
          client.send(JSON.stringify(payload));
        }
      } catch (err) {
        console.warn('WebSocket broadcast error:', err.message);
      }
    });
  }

  if (overlayWindow && !overlayWindow.isDestroyed()) {
    try {
      overlayWindow.webContents.send('ai-update', payload);
    } catch (err) {
      console.warn('Overlay broadcast error:', err.message);
    }
  }
}

/* ============================================================
   DIRECT VOICE STATE SYNC
============================================================ */
function setDirectVoiceState(enabled, reason = '') {
  directVoiceEnabled = Boolean(enabled);
  console.log(`🎙️ Direct Voice: ${directVoiceEnabled ? 'ON' : 'OFF'}${reason ? ` | ${reason}` : ''}`);

  if (overlayWindow && !overlayWindow.isDestroyed()) {
    try {
      overlayWindow.webContents.send('direct-voice-state', directVoiceEnabled);
    } catch {}
  }

  broadcast({
    type: 'direct-voice-state',
    enabled: directVoiceEnabled
  });
}

/* ============================================================
   CLEAN AI RESPONSE
============================================================ */
function cleanAIResponse(rawText) {
  if (!rawText) return '';
  let text = String(rawText).trim();

  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  text = text.replace(/&lt;think&gt;[\s\S]*?&lt;\/think&gt;/gi, '');
  text = text.replace(/<think>[\s\S]*$/gi, '');

  if (text.includes('```')) {
    return text.trim();
  }

  const metaKeywords = [
    'total words', 'word count', 'thinking process', 'deconstruct question',
    'internal reasoning', 'analysis:', 'reasoning:', 'draft:', 'refine:',
    'strategy:', 'constraints:', 'chain of thought:', 'thought process:',
    'let me think', 'i need to analyze'
  ];

  const lines = text.split('\n');
  const cleanedLines = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (cleanedLines.length > 0 && cleanedLines[cleanedLines.length - 1] !== '') {
        cleanedLines.push('');
      }
      continue;
    }

    const normalized = trimmed.replace(/^[-*•#>\s]+/, '').toLowerCase();
    const isMeta = metaKeywords.some(keyword => normalized.startsWith(keyword));

    if (!isMeta) {
      cleanedLines.push(trimmed);
    }
  }

  return cleanedLines.join('\n').trim();
}

/* ============================================================
   RESPONSE VALIDATION
============================================================ */
function isValidAIResponse(text) {
  if (!text) return false;
  const cleaned = text.trim();
  if (cleaned.length < 3) return false;

  const invalidResponses = [
    'i cannot answer', "i can't answer", 'unable to answer',
    'no answer', 'empty response', 'as an ai', 'i am an ai'
  ];

  return !invalidResponses.some(phrase => cleaned.toLowerCase() === phrase);
}

/* ============================================================
   GEMINI API MULTI-KEY ROTATION
============================================================ */
async function callGemini({ prompt, images = [] }) {
  if (geminiKeys.length === 0) throw new Error('No Gemini API keys configured.');

  for (const modelName of GEMINI_MODELS) {
    for (let i = 0; i < geminiKeys.length; i++) {
      const keyIndex = (currentGeminiKeyIndex + i) % geminiKeys.length;
      const key = geminiKeys[keyIndex];

      try {
        console.log(`Gemini [${modelName}] Key #${keyIndex + 1}`);
        const ai = new GoogleGenAI({ apiKey: key });
        const contents = images.length > 0 ? [prompt, ...images] : [prompt];

        const responsePromise = ai.models.generateContent({
          model: modelName,
          contents
        });

        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Gemini call timed out (15s)')), 15000)
        );

        const response = await Promise.race([responsePromise, timeoutPromise]);
        const answer = cleanAIResponse(response?.text || '');

        if (isValidAIResponse(answer)) {
          currentGeminiKeyIndex = (keyIndex + 1) % geminiKeys.length;
          return answer;
        }
      } catch (err) {
        console.warn(`Gemini [${modelName}] Key #${keyIndex + 1} failed:`, err?.message || err);
      }
    }
  }

  throw new Error('All Gemini API keys and models exhausted.');
}

/* ============================================================
   GROQ REQUESTS
============================================================ */
async function callGroqText(prompt) {
  if (!groq) throw new Error('GROQ_API_KEY is not configured.');

  for (const modelName of GROQ_VOICE_MODELS) {
    try {
      console.log(`Groq Text -> ${modelName}`);
      const chatCompletion = await groq.chat.completions.create({
        model: modelName,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        max_completion_tokens: TEXT_MAX_TOKENS
      });

      const answer = cleanAIResponse(chatCompletion?.choices?.[0]?.message?.content || '');
      if (isValidAIResponse(answer)) return answer;
    } catch (err) {
      console.warn(`Groq ${modelName} failed:`, err?.message || err);
    }
  }

  throw new Error('All Groq text models failed.');
}

async function callGroqVision({ prompt, images }) {
  if (!groq) throw new Error('GROQ_API_KEY is not configured.');

  const groqImageContent = images.map(img => ({
    type: 'image_url',
    image_url: {
      url: `data:${img.inlineData.mimeType};base64,` + img.inlineData.data
    }
  }));

  for (const modelName of GROQ_VISION_MODELS) {
    try {
      console.log(`Groq Vision -> ${modelName}`);
      const chatCompletion = await groq.chat.completions.create({
        model: modelName,
        messages: [{
          role: 'user',
          content: [{ type: 'text', text: prompt }, ...groqImageContent]
        }],
        temperature: 0.1,
        max_completion_tokens: VISION_MAX_TOKENS
      });

      const answer = cleanAIResponse(chatCompletion?.choices?.[0]?.message?.content || '');
      if (isValidAIResponse(answer)) return answer;
    } catch (err) {
      console.warn(`Groq Vision ${modelName} failed:`, err?.message || err);
    }
  }

  throw new Error('All Groq Vision models failed.');
}

/* ============================================================
   FAST PIPELINE ENGINE
============================================================ */
async function executeVisionFast({ prompt, images }) {
  if (isGeminiEnabled) {
    try {
      return await callGemini({ prompt, images });
    } catch (geminiError) {
      console.warn('Gemini Vision failed -> Switching to Groq Vision backup...');
      broadcast({ type: 'status', msg: 'Gemini exhausted -> Using Groq Vision...' });
    }
  } else {
    console.log('⚡ Gemini OFF: Routing Vision directly to Groq...');
  }

  try {
    return await callGroqVision({ prompt, images });
  } catch (groqError) {
    throw new Error('All Vision AI endpoints exhausted.');
  }
}

async function executeTextWithFallback(prompt) {
  if (!isGeminiEnabled) {
    console.log('⚡ Gemini OFF: Routing Text purely through Groq...');
    return await callGroqText(prompt);
  }

  try {
    return await callGroqText(prompt);
  } catch (groqError) {
    console.warn('Groq text pool exhausted -> Switching to Gemini backup...');
    broadcast({ type: 'status', msg: 'Groq exhausted -> Using Gemini backup...' });
  }

  try {
    return await callGemini({ prompt });
  } catch (geminiError) {
    throw new Error('All Voice/Text AI endpoints exhausted.');
  }
}

/* ============================================================
   RESUME FILE LOADING
============================================================ */
function loadResumeFile(filePath) {
  try {
    if (!fs.existsSync(filePath)) return false;

    const ext = path.extname(filePath).toLowerCase();
    if (ext === '.pdf') {
      broadcast({ type: 'status', msg: '⚠️ Select JSON/TXT/MD Resume' });
      return false;
    }

    const raw = fs.readFileSync(filePath, 'utf-8');
    try {
      const parsed = JSON.parse(raw);
      resumeProfileText = JSON.stringify(parsed, null, 2);
    } catch {
      resumeProfileText = raw.slice(0, 5000).trim();
    }

    console.log(`✅ Resume loaded: ${filePath}`);
    broadcast({ type: 'status', msg: '✅ Resume Context Synced' });
    return true;
  } catch (err) {
    console.error('Failed to load resume:', err);
    return false;
  }
}

loadResumeFile(path.join(__dirname, 'resumeContext.json'));

/* ============================================================
   DYNAMIC INTERVIEW PROMPT
============================================================ */
function getModePrompt() {
  const profileSnippet = resumeProfileText
    ? `
CANDIDATE PROFILE:
${resumeProfileText}

Use this profile ONLY when the interviewer asks about candidate's experience, projects, responsibilities, or skills.
Never invent experience, companies, or technologies.
`
    : '';

  if (currentMode === 'code') {
    return `
You are an expert live interview coding assistant.
${profileSnippet}

Analyze the question carefully:
- If asked to write code/program/query/algorithm: return ONLY the actual working code in a markdown code block.
- Identify the requested language (C#, Java, Python, JavaScript, SQL, etc.). Never assume C# unless required.
- Do not provide conversational filler or explanations outside code.
- For algorithms include:
  // Time: O(...)
  // Space: O(...)
`;
  }

  return `
You are an expert live interview assistant.
${profileSnippet}

The interviewer can ask ANY technical or HR question:
- If asked for CODE / QUERY / PROGRAM: provide the actual code/query directly.
- If asked "WHAT IS..." or "EXPLAIN": give a crisp conceptual explanation starting with official definition.
- If COMPARISON: compare directly focusing on practical project trade-offs.
- If ARCHITECTURE / SCENARIO: explain approach, decisions, and best practices.
- If PERSONAL EXPERIENCE: use first-person language based ONLY on candidate profile.
- Length: 2-3 points for simple, 3-5 points for normal, detailed for complex architectures.
- Do NOT output reasoning, <think> tags, analysis, or meta commentary.
- Start directly with the answer.
`;
}

/* ============================================================
   SCREEN CAPTURE
============================================================ */
async function captureScreen() {
  try {
    const started = Date.now();
    const rawBuffer = await screenshot({ format: 'jpg' });

    const optimizedBuffer = await sharp(rawBuffer)
      .resize({ width: 1280, fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 60, mozjpeg: true })
      .toBuffer();

    imageBuffer = [{
      inlineData: {
        data: optimizedBuffer.toString('base64'),
        mimeType: 'image/jpeg'
      }
    }];

    console.log(`📸 Screenshot ready: ${Math.round(optimizedBuffer.length / 1024)} KB | ${((Date.now() - started) / 1000).toFixed(2)}s`);
    broadcast({ type: 'status', msg: '📸 Screenshot Captured' });
  } catch (err) {
    console.error('Capture error:', err);
    broadcast({ type: 'status', msg: 'Screenshot capture failed.' });
  }
}

/* ============================================================
   VISION SOLVE
============================================================ */
async function solveVisionQuestion() {
  if (isVisionBusy) {
    broadcast({ type: 'status', msg: '⏳ Vision analysis already running...' });
    return;
  }

  if (imageBuffer.length === 0) {
    broadcast({ type: 'status', msg: 'No screenshot! Press Snap first' });
    return;
  }

  isVisionBusy = true;
  broadcast({ type: 'status', msg: `⚡ Analyzing ${currentMode.toUpperCase()}...` });

  const currentImages = [...imageBuffer];
  imageBuffer = [];

  const prompt = `
Read the interview question/problem visible in the screenshot.
Ignore background UI and answer directly.
${getModePrompt()}
`;

  const startTime = performance.now();

  try {
    const answer = await executeVisionFast({ prompt, images: currentImages });
    const elapsedMs = performance.now() - startTime;
    const durationSec = (elapsedMs / 1000).toFixed(2);

    lastAnswerText = answer;
    broadcast({
      type: 'answer',
      text: answer,
      duration: `${durationSec}s`
    });

    console.log(`🏁 Complete Vision: ${durationSec}s`);
  } catch (err) {
    console.error('Vision error:', err);
    broadcast({ type: 'status', msg: 'Vision Error: ' + err.message });
  } finally {
    setTimeout(() => { isVisionBusy = false; }, 400);
  }
}

/* ============================================================
   INTERNAL AUDIO PROCESSING
============================================================ */
ipcMain.on('process-internal-audio', async (event, base64Audio) => {
  if (isAudioBusy || isVisionBusy) {
    console.log('Skipping audio: Worker busy');
    return;
  }

  isAudioBusy = true;
  const startTime = performance.now();
  broadcast({ type: 'status', msg: '🎙️ Transcribing speech...' });

  try {
    if (!groq) throw new Error('GROQ_API_KEY is not configured.');

    const audioBuffer = Buffer.from(base64Audio, 'base64');
    const audioFile = await toFile(audioBuffer, 'audio.webm', { type: 'audio/webm' });

    const transcription = await groq.audio.transcriptions.create({
      file: audioFile,
      model: WHISPER_MODEL,
      language: 'en'
    });

    const questionText = transcription?.text?.trim();

    if (!questionText || questionText.length < 4) {
      broadcast({ type: 'status', msg: 'No clear speech detected.' });
      return;
    }

    console.log(`🎙️ Heard: ${questionText}`);
    broadcast({
      type: 'status',
      msg: `Heard: "${questionText.slice(0, 45)}..." -> Thinking...`
    });

    const prompt = `
INTERVIEWER QUESTION:
"${questionText}"

${getModePrompt()}
Now provide the final answer directly.
`;

    const answer = await executeTextWithFallback(prompt);
    const elapsedMs = performance.now() - startTime;
    const durationSec = (elapsedMs / 1000).toFixed(2);

    lastAnswerText = answer;
    broadcast({
      type: 'answer',
      text: `🎙️ "${questionText}"\n\n💡 ${answer}`,
      duration: `${durationSec}s`
    });

    console.log(`🏁 Complete Voice: ${durationSec}s`);
  } catch (err) {
    console.error('Audio processing error:', err);
    broadcast({ type: 'status', msg: 'Audio Error: ' + err.message });
  } finally {
    isAudioBusy = false;
  }
});

/* ============================================================
   ELECTRON IPC & LIFECYCLE
============================================================ */
ipcMain.handle('get-desktop-source-id', async () => {
  const sources = await desktopCapturer.getSources({ types: ['screen'] });
  return sources[0]?.id;
});

ipcMain.handle('select-resume-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    title: 'Select Resume / Profile File',
    filters: [
      { name: 'Supported Files', extensions: ['json', 'txt', 'md'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    properties: ['openFile']
  });

  if (!canceled && filePaths.length > 0) {
    const success = loadResumeFile(filePaths[0]);
    return {
      success,
      path: filePaths[0],
      name: path.basename(filePaths[0])
    };
  }
  return { success: false };
});

ipcMain.on('switch-mode', (event, mode) => {
  currentMode = mode;
  broadcast({ type: 'status', msg: `Mode: ${mode.toUpperCase()}` });
});

ipcMain.on('toggle-gemini', (event, enabled) => {
  isGeminiEnabled = Boolean(enabled);
  console.log(`✨ Gemini Engine: ${isGeminiEnabled ? 'ENABLED' : 'DISABLED (Groq Only)'}`);
  broadcast({
    type: 'status',
    msg: isGeminiEnabled ? 'Engine: Gemini + Groq' : 'Engine: Pure Groq Only'
  });
});

/* ============================================================
   FIXED SIZE WINDOW DRAG HANDLER (NO EXPANDING BUG)
============================================================ */
// ipcMain.on('window-drag-move', (event, { mouseX, mouseY }) => {
//   if (!overlayWindow) return;
//   const { x, y } = screen.getCursorScreenPoint();
//   const [currentWidth, currentHeight] = overlayWindow.getSize();

//   overlayWindow.setBounds({
//     x: Math.round(x - mouseX),
//     y: Math.round(y - mouseY),
//     width: currentWidth,
//     height: currentHeight
//   });
// });
ipcMain.on('window-drag-move', (event, { mouseX, mouseY }) => {
  if (!overlayWindow || overlayWindow.isDestroyed()) return;

  const { x, y } = screen.getCursorScreenPoint();
  const [w, h] = overlayWindow.getSize();

  // సైజ్ లాక్ చేసి కేవలం పొజిషన్ మాత్రమే మూవ్ చేయడం
  overlayWindow.setBounds({
    x: Math.round(x - mouseX),
    y: Math.round(y - mouseY),
    width: w,
    height: h
  }, false);
});
ipcMain.on('direct-voice-changed', (event, enabled) => {
  setDirectVoiceState(enabled, 'Overlay toggle');
});

ipcMain.on('toggle-direct-voice', () => {
  setDirectVoiceState(!directVoiceEnabled, 'Main toggle');
});

ipcMain.on('window-minimize', () => overlayWindow?.minimize());
ipcMain.on('window-toggle-maximize', () => {
  if (!overlayWindow) return;
  overlayWindow.isMaximized() ? overlayWindow.unmaximize() : overlayWindow.maximize();
});
ipcMain.on('window-hide', () => overlayWindow?.hide());

function createOverlayWindow() {
  const { width } = screen.getPrimaryDisplay().workAreaSize;

  overlayWindow = new BrowserWindow({
    width: 520,
    height: 460,
    x: width - 540,
    y: 30,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    focusable: true,
    backgroundColor: '#00000000',
    type: 'toolbar',
    hasShadow: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  overlayWindow.setAlwaysOnTop(true, 'screen-saver');
  overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  overlayWindow.on('blur', () => {
    overlayWindow.setAlwaysOnTop(true, 'screen-saver');
  });

  overlayWindow.on('move', () => {
    overlayWindow.setAlwaysOnTop(true, 'screen-saver');
  });

  overlayWindow.on('resize', () => {
    overlayWindow.setAlwaysOnTop(true, 'screen-saver');
  });

  overlayWindow.setContentProtection(true);
  overlayWindow.loadFile(path.join(__dirname, 'overlay.html'));

  overlayWindow.webContents.once('did-finish-load', () => {
    try {
      overlayWindow.webContents.send('direct-voice-state', directVoiceEnabled);
      overlayWindow.webContents.send('gemini-state', isGeminiEnabled);
    } catch {}
  });

  overlayWindow.on('closed', () => { overlayWindow = null; });
}

/* ============================================================
   GLOBAL HOTKEYS
============================================================ */
uIOhook.on('keydown', event => {
  if (!event.ctrlKey || !event.shiftKey) return;

  const isKey1 = event.keycode === UiohookKey.N1 || event.keycode === UiohookKey.Num1 || event.rawcode === 49 || event.rawcode === 97 || event.rawcode === 33;
  const isKey2 = event.keycode === UiohookKey.N2 || event.keycode === UiohookKey.Num2 || event.rawcode === 50 || event.rawcode === 98 || event.rawcode === 64;

  if (isKey1) {
    currentMode = 'code';
    broadcast({ type: 'status', msg: 'Mode: CODING / DYNAMIC' });
    return;
  }

  if (isKey2) {
    currentMode = 'speech';
    broadcast({ type: 'status', msg: 'Mode: SPEECH / DYNAMIC' });
    return;
  }

  if (event.keycode === UiohookKey.C) {
    setDirectVoiceState(false, 'Screenshot capture');
    setTimeout(() => { captureScreen(); }, 50);
    return;
  }

  if (event.keycode === UiohookKey.S) {
    setDirectVoiceState(false, 'Vision solve');
    setTimeout(() => { solveVisionQuestion(); }, 50);
    return;
  }

  if (event.keycode === UiohookKey.V) {
    overlayWindow?.webContents.send('toggle-voice');
    return;
  }

  if (event.keycode === UiohookKey.D) {
    setDirectVoiceState(!directVoiceEnabled, 'Ctrl + Shift + D');
    return;
  }

  if (event.keycode === UiohookKey.X) {
    imageBuffer = [];
    broadcast({ type: 'clear' });
    broadcast({ type: 'status', msg: 'Cleared.' });
    return;
  }

  if (event.keycode === UiohookKey.H) {
    overlayWindow?.isVisible() ? overlayWindow.hide() : overlayWindow?.show();
    return;
  }

  if (event.keycode === UiohookKey.Q && lastAnswerText) {
    clipboard.writeText(lastAnswerText);
    broadcast({ type: 'status', msg: '📋 Copied to Clipboard!' });
    return;
  }
});

/* ============================================================
   APP LIFECYCLE
============================================================ */
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
    server.listen(3000, '0.0.0.0', () => {
      console.log('🚀 Stealth Assistant running on port 3000');
    });
  });
}

app.on('before-quit', () => {
  try { uIOhook.stop(); } catch {}
  try { server.close(); } catch {}
});

app.on('window-all-closed', () => {
  try { uIOhook.stop(); } catch {}
  if (process.platform !== 'darwin') app.quit();
});