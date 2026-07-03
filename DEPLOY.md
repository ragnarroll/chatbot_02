# Deploying your chatbot to the web

This gets your `rag_chatbot.py` logic running as a live web server, and a
chat bubble on your actual website talking to it. No command-line git
required — everything below uses web browser UIs.

## Part 1 — Put the server code on GitHub

1. Go to https://github.com and create a free account if you don't have one.
2. Click the **+** in the top right → **New repository**. Name it e.g.
   `rag-chatbot-server`. Keep it Public (needed for Render's free tier).
   Click **Create repository**.
3. On the new repo page, click **"uploading an existing file"**.
4. Drag in the 3 files from the `server/` folder: `main.py`,
   `requirements.txt`, and `Procfile`. Commit the upload.

## Part 2 — Deploy it on Render

1. Go to https://render.com and sign up (you can sign up with your GitHub
   account, which makes step 3 easier).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if prompted, then select the
   `rag-chatbot-server` repo you just created.
4. Fill in:
   - **Name**: anything, e.g. `rag-chatbot`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
5. Under **Environment Variables**, add:
   - `ANTHROPIC_API_KEY` = your key from https://console.anthropic.com
6. Click **Create Web Service**. Render will build and deploy — takes a
   few minutes the first time (downloading the embedding model).
7. When it's live, Render gives you a URL like:
   `https://rag-chatbot-xxxx.onrender.com`

Test it worked by visiting that URL in your browser — you should see
`{"status": "ok"}`.

**Free tier note:** Render's free instances "sleep" after 15 minutes of no
traffic, and take ~30-60 seconds to wake up on the next request. Fine for
testing; if this matters for real customers, upgrade to a paid instance
($7/mo) later — no code changes needed.

## Part 3 — Lock down access (CORS)

Before customers use it, edit `main.py`:

```python
ALLOWED_ORIGINS = [
    "https://your-actual-domain.com",   # <-- your real website
]
```

Re-upload the edited file to GitHub (same drag-and-drop as before, this
time using **"Add file" → "Upload files"** on the existing repo) — Render
will auto-redeploy.

Without this, anyone could technically call your API from any website,
running up your Anthropic API bill.

## Part 4 — Add the widget to your website

1. Open `chat-widget.html` in a text editor.
2. Change this line to your real Render URL from Part 2, with `/chat` on
   the end:
   ```
   const API_URL = "https://rag-chatbot-xxxx.onrender.com/chat";
   ```
3. Copy the entire file's contents.
4. In your website builder:
   - **Squarespace**: Business plan or higher → Page → add a **Code Block**
     → paste it in. Or Settings → Advanced → Code Injection → Footer, to
     show it site-wide.
   - **WordPress**: add a **Custom HTML block** on a page, or use a plugin
     like "Insert Headers and Footers" to add it site-wide.
5. Publish. You should see the chat bubble in the bottom-right corner.

## Updating what the bot knows

Edit the `documents = [...]` list in `main.py`, re-upload to GitHub, Render
redeploys automatically. For anything beyond a handful of hardcoded
sentences (a real knowledge base), the natural next step is loading from
actual files instead — happy to build that when you're ready.

## Costs to expect

- **Render**: free while testing; $7/mo if you outgrow the free tier's
  sleep behavior
- **Anthropic API**: pay-per-use, billed separately from Claude.ai — a few
  cents per conversation at this scale
- **Hugging Face model download**: free
