# ✨ Image Enhancer v2 — Online (Streamlit version)

AI photo enhancer — **4x bara + saaf**, free, unlimited, browser mein chalta hai.

- **AI model:** Real-ESRGAN (general-x4v3) — smooth, clean upscales
- **Gentle pipeline:** halki denoise → 4x upscale → bohat halki sharpening
- **Output:** 6000px tak (lambi side), PNG/JPG download

## Deploy (free, ~5-10 minute — torch download ki wajah se pehli dafa thora slow)

Tum yeh flow jantay ho:

1. GitHub par **nayi repository** banao (misal `image-enhancer-v2`) — ya purani wali mein files replace kar do
2. Is folder ki **saari files + `models/` folder** upload karo (models folder lazmi!)
3. **share.streamlit.io** kholo → GitHub se sign in → **"New app"**
4. Repository + branch (`main`) + `app.py` select karo → **Deploy**
5. 5-10 minute mein link taiyaar 🔖 (pehli dafa torch install hota hai, is liye time lagta hai)

**2 chhoti baatein:**
- Free CPU par bari photo (4000px) enhance honay mein **1-3 minute** lag sakta hai — normal hai, wait karo ☕
- Free apps 12 ghante idle rehne par so jati hain (~30 sec mein jaag jati hain)

## Honest limits (zaroor parho)

- AI **4x tak** asli detail banata hai — is se zyada "enhance" sirf bara-blurry image hota hai
- Output **6000px** tak capped hai (free server ki memory limit)
- **16K** jaise daway jhoot hotay hain — yeh real, tested enhancer hai, jadoo nahi 🙂
