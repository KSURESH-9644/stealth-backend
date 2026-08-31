import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { GoogleGenAI } from '@google/genai';
import Groq, { toFile } from 'groq-sdk';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

// High payload limits for Base64 screenshots and audio buffers
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Dynamic Resume Context Loading (JSON, TXT, MD, etc.)
let resumeProfileText = '';
function loadResumeContext(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      const raw = fs.readFileSync(filePath, 'utf-8');
      try {
        const parsed = JSON.parse(raw);
        resumeProfileText = JSON.stringify(parsed, null, 2);
      } catch {
        resumeProfileText = raw;
      }
      console.log(`✅ Resume Context loaded from: ${filePath}`);
      return true;
    }
  } catch (err) {
    console.error('Failed to load resume context:', err);
  }
  return false;
}

// Initial Resume Load
loadResumeContext(path.join(__dirname, 'resumeContext.json'));

// API Keys Setup
const rawKeys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const geminiKeys = rawKeys.split(',').map(k => k.trim()).filter(Boolean);
let currentKeyIndex = 0;
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// WebSocket Real-time Broadcasting
let clients = [];
wss.on('connection', (ws) => {
  clients.push(ws);
  ws.send(JSON.stringify({ type: 'status', msg: 'Connected to Cloud Assistant' }));
  ws.on('close', () => { clients = clients.filter(c => c !== ws); });
});

function broadcast(payload) {
  clients.forEach(c => {
    if (c.readyState === 1) c.send(JSON.stringify(payload));
  });
}

// Mode Prompts
function getModePrompt(mode = 'speech') {
  const profileSnippet = resumeProfileText
    ? `\n[CANDIDATE PROFILE & STACK]:\n${resumeProfileText}\n`
    : '';

  if (mode === 'code') {
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

// Smart AI Routing: Vision -> Gemini First | Voice -> Groq First
async function executeWithFallback({ prompt, images = [], isVision = false }) {
  // 1️⃣ VISION / SCREENSHOTS -> GEMINI FIRST (Groq Backup)
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

  // 2️⃣ VOICE / AUDIO QUESTIONS -> GROQ FIRST (Gemini Backup)
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

// API: Process Voice Audio
app.post('/api/solve-audio', async (req, res) => {
  const { base64Audio, mode = 'speech' } = req.body;
  if (!base64Audio) return res.status(400).json({ error: 'Missing audio payload' });

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
      return res.json({ status: 'No clear speech detected' });
    }

    broadcast({ type: 'status', msg: `Heard: "${questionText.slice(0, 25)}..." -> Thinking...` });
    const prompt = getModePrompt(mode) + `\n\nInterviewer Question: "${questionText}"\nProvide the direct response:`;
    const answer = await executeWithFallback({ prompt, isVision: false });

    const payload = { type: 'answer', text: `🎙️ "${questionText}"\n\n💡 ${answer}` };
    broadcast(payload);
    res.json(payload);
  } catch (err) {
    console.error('Audio processing error:', err);
    broadcast({ type: 'status', msg: 'Audio Error: ' + err.message });
    res.status(500).json({ error: err.message });
  }
});

// API: Process Screen Vision
app.post('/api/solve-vision', async (req, res) => {
  const { imageBase64, mode = 'speech' } = req.body;
  if (!imageBase64) return res.status(400).json({ error: 'Missing image payload' });

  broadcast({ type: 'status', msg: `Analyzing screen [${mode.toUpperCase()}]...` });
  try {
    const prompt = getModePrompt(mode) + "\nAnalyze this technical problem on screen and solve it directly:";
    const images = [{ inlineData: { data: imageBase64, mimeType: 'image/png' } }];
    const answer = await executeWithFallback({ prompt, images, isVision: true });

    const payload = { type: 'answer', text: answer };
    broadcast(payload);
    res.json(payload);
  } catch (err) {
    console.error('Vision processing error:', err);
    broadcast({ type: 'status', msg: 'Vision Error: ' + err.message });
    res.status(500).json({ error: err.message });
  }
});

// API: Sync Dynamic Resume Content
app.post('/api/upload-resume', (req, res) => {
  const { content } = req.body;
  if (content) {
    resumeProfileText = content;
    broadcast({ type: 'status', msg: '✅ Resume Context Updated' });
    return res.json({ success: true, msg: 'Resume updated' });
  }
  res.status(400).json({ error: 'Empty content' });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Stealth Assistant Backend running on port ${PORT}`);
});