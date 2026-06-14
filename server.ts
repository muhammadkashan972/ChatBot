import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// Lazily initialize Gemini AI client
let aiClient: GoogleGenAI | null = null;

function getAiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is not defined. Please verify it is added under Settings > Secrets.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Middleware to parse incoming JSON payloads
  app.use(express.json());

  // API endpoint for streaming chatbot responses
  app.post("/api/chat/stream", async (req, res) => {
    try {
      const { messages } = req.body;
      if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: "Messages array is required." });
      }

      const client = getAiClient();

      // Restructure messages to match Google GenAI SDK expectation
      // Gemini models expect 'user' and 'model' roles, and allow multiple parts (text, image, video).
      const contents = messages.map((msg: any) => {
        const parts: any[] = [];
        
        // Always add the textual content
        parts.push({ text: msg.content || "" });

        // If an attachment is specified, map it to the inlineData scheme
        if (msg.attachment && msg.attachment.base64 && msg.attachment.mimeType) {
          parts.push({
            inlineData: {
              mimeType: msg.attachment.mimeType,
              data: msg.attachment.base64
            }
          });
        }

        return {
          role: msg.role === "assistant" ? "model" : "user",
          parts: parts
        };
      });

      // Set headers for Chunked Transmission / Streaming
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Transfer-Encoding', 'chunked');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const systemInstruction = 
        "Your name is 'Kashan Chatbot', a professional, highly intelligent, and exceptionally friendly AI assistant. " +
        "You help users with programming/coding, web development, studies/academics, content/copy writing, technology, business ideas, trading, history, and general inquiries. " +
        "\n\n" +
        "YOUR CORE GOALS:\n" +
        "1. Provide accurate and helpful answers. Admit uncertainty clearly or say 'Mujhe is baare mein pakka nahi pata' instead of fabricating facts.\n" +
        "2. Respond quickly, clearly, with a professional yet warm and friendly tone.\n" +
        "3. Format answers beautifully using clean headers, bullet points, bold text, and code blocks where helpful (always specify the language for syntax highlighting, e.g., ```typescript).\n" +
        "4. Explain complex topics in simple terms and offer step-by-step guidance when solving problems.\n" +
        "5. Keep responses robust but concise unless a detailed step-by-step explanation is requested.\n\n" +
        "CRITICAL LANGUAGE DIRECTIVE:\n" +
        "You MUST reply 100% in the exact language context matching the user's prompt. " +
        "If they message you in Hindi, reply in clean and natural Hindi. " +
        "If they message you in Urdu, reply in clean and natural Urdu. " +
        "If they write in a mixture of Urdu/Hindi and English (Roman Urdu/Hindi/Hinglish like 'reply kar do' or 'kaise ho'), reply in that exact natural conversational Hinglish/Roman-Urdu tone! " +
        "If they write in Arabic, Spanish, French, German, or English, adapt instantaneously and answer perfectly in that language.\n\n" +
        "DYNAMIC FOLLOW-UP QUESTIONS DIRECTIVE:\n" +
        "At the absolute end of your response, you MUST suggest 2 or 3 highly relevant, short, clicking-focused follow-up questions that the user might want to ask next. " +
        "Format these follow-up questions strictly inside '[FOLLOW_UP]' and '[/FOLLOW_UP]' brackets at the very bottom, with each question on a new line starting with a dash, exactly like this:\n" +
        "[FOLLOW_UP]\n" +
        "- Can you give me an example of this code?\n" +
        "- What is the primary risk of this approach?\n" +
        "[/FOLLOW_UP]\n" +
        "Make sure to use the exact matching language / tone (Urdu/Hindi/English/Hinglish) for these follow-up suggestions as well!";

      const responseStream = await client.models.generateContentStream({
        model: "gemini-3.5-flash",
        contents: contents,
        config: {
          systemInstruction: systemInstruction,
        }
      });

      for await (const chunk of responseStream) {
        if (chunk.text) {
          res.write(chunk.text);
        }
      }
      res.end();

    } catch (error: any) {
      console.error("Error generating stream:", error);
      if (res.headersSent) {
        res.write(`\n\n[Error: ${error.message || "An issue occurred while streaming"}]`);
        res.end();
      } else {
        res.status(500).json({ error: error.message || "Could not generate AI response." });
      }
    }
  });

  // Vite middleware block for development and production build serving
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server started running on host 0.0.0.0 on port ${PORT}`);
  });
}

startServer();
