# XMIDAS - Windows Installation Guide

This guide will walk Windows users through installing and running the **XMIDAS** (X-ray Multimodal Image Data Analysis Software) application from source.

---

## 📦 Requirements

* [Miniconda or Anaconda (64-bit)](https://docs.conda.io/en/latest/miniconda.html)
* (Optional) [Git for Windows](https://git-scm.com/download/win)

---

## 🧰 Step-by-Step Installation

### 1. Clone or Download the Repository

**Option A – With Git (recommended):**

```bash
git clone https://github.com/pattammattel/xmidas.git
cd xmidas
```

**Option B – Without Git:**

1. Visit: [https://github.com/pattammattel/xmidas](https://github.com/pattammattel/xmidas)
2. Click **Code → Download ZIP**
3. Extract ZIP
4. **Open Anaconda Prompt and navigate into the extracted `xmidas/` folder**:

   ```bash
   cd path\to\extracted\xmidas
   ```

🚨 **IMPORTANT:** You **must** be inside the `xmidas` folder before continuing! All commands in the next steps assume this location.

---

### 2. Create a Conda Environment

```bash
conda create -n xmidas-env python=3.12
conda activate xmidas-env
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


---

### 4. Run XMIDAS

```bash
python -m xmidas.main
```


---

## 📌 Troubleshooting

* ✅ **Double-check that you're in the `xmidas/` folder** before running anything.
* If you see `ModuleNotFoundError: No module named 'xmidas'`, you are probably in the wrong directory.
* If PyQt-related errors occur, ensure `PyQt6` is installed and compatible with Python 3.12.

---

## 🗂 Optional: Create a Desktop Shortcut (Windows only)

Inside `MiscFiles-FindProperLocation/`, you’ll find a helper script:

```bash
python create_shortcut.py
```

This will generate a Windows `.bat` shortcut to launch the GUI.

---

## 🔗 Resources

* 📖 Documentation: See `docs/source/index.rst`
* 🐞 Issues / Bugs: [GitHub Issues](https://github.com/pattammattel/xmidas/issues)

---

Happy analyzing!

*– XMIDAS Development Team*
