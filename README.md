# 🛒 AI-Powered E-Commerce Catalog Generator

> Transform product keywords into compelling listings using **Llama 3 8B** model

## ✨ Features

- 🤖 **AI-Generated Content** - Automatic product titles and descriptions
- 🎯 **Smart Validation** - Category-feature matching
- ⚠️ **Warning System** - Alerts for mismatched categories
- 🔄 **Auto Form Filling** - Seamless field population

---

## 🚀 Quick Setup

### Step 1: Install Ollama

**Windows:**
- Download from [https://ollama.ai/download](https://ollama.ai/download)

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Download AI Model

```bash
ollama pull llama3:8b
```

### Step 3: Clone & Setup

```bash
# Clone repository
git clone https://github.com/Aarsh59/Meesho-Dice-Challenge-2.0



# Activate virtual environment
python3 -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install flask flask-cors
```

### Step 4: Run the Application

**Start Backend:**
```bash
python Flask.py
```

**Open Frontend:**
- Open `front.html` in your browser
- Or serve locally: `python -m http.server 3000`

---

## 📱 How It Works

### 1. Backend Running
*[Screenshot of Flask server successfully starting]*

![Backend Running](images/image2.png)


### 2. User Input
*[Screenshot of user entering product features]*

![User Input](images/image3.png)

### 3. Category Mismatch Warning
*[Screenshot showing warning when features don't match category]*

![Mismatch Warning](images/image4.png)



### 4. Successful Generation
*[Screenshot showing successful AI generation message]*

![Successful Generation](images/image5.png)


---

## 🎯 Usage

1. **Select Category** - Choose from dropdown (Electronics, Clothing, etc.)
2. **Enter Features** - Type product keywords (e.g., "16GB RAM, RTX 4060")
3. **Generate** - Click 🪄 magic button
4. **Review** - Check auto-filled title and description

## 📁 Project Structure

```
📦 project/
├── 📄 Flask.py          # Backend server
├── 📄 front.html        # Frontend interface  
├── 📄 README.md         # This file
└── 📁 images/           # Demo screenshots
    ├── image2.png
    ├── image3.png
    ├── image4.png
    └── image5.png
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

**🎉 Start generating AI-powered product listings now!**
