# Deploy the demo

The repository is ready for a Hugging Face Space, but the final Space must be created from your own Hugging Face account. The demo uses five exact commands: `squat down`, `shoot the target`, `miss the target`, `dunk it`, and `drift`.

## Option 1: Hugging Face Spaces

### Create the account and Space

1. Open <https://huggingface.co/join> in a web browser.
2. Create a free account and confirm the email address Hugging Face sends you.
3. While signed in, open <https://huggingface.co/new-space>.
4. Enter `multi-task-rl-demo` as the Space name.
5. Select **Gradio** as the SDK.
6. Select **Public** visibility and the free **CPU basic** hardware.
7. Create the Space. Leave the generated starter files alone for now.
8. Open <https://huggingface.co/settings/tokens> in a new browser tab.
9. Create a token with **write** permission. Copy it and keep it private. This token is used as the password in step 16; never put it in a file or paste it into chat.

### Upload this repository from a terminal

10. Install Git from <https://git-scm.com/downloads> if the `git` command is not already available.
11. Open Terminal on macOS/Linux or PowerShell on Windows.
12. Copy and paste these commands one line at a time:

```bash
git clone https://github.com/fymunotbready-ctrl/multi-task-rl.git
cd multi-task-rl
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/multi-task-rl-demo
git push --force space main
```

13. Before running the third command, replace `YOUR_HF_USERNAME` with the username created in step 2. Do not change the Space name unless you chose a different name in step 4.
14. The final command replaces only the starter commit created with the new Space. It uploads the repository's `main` branch, including the model checkpoints and demo assets.
15. When Git asks for a username, enter your Hugging Face username.
16. When Git asks for a password, paste the write token from step 9. The terminal may show no characters while you paste; press Enter once.
17. Return to the Space page. Open **Logs** if the page says **Building**. The first build installs `requirements.txt` and can take several minutes.
18. Wait until the Space says **Running**, then enter one of the five exact commands and select **Run command**.

The repository's `README.md` contains the required Spaces metadata: Gradio SDK 5.49.1 and `app.py` as the entry point. No secrets or paid hardware are required.

## Option 2: run locally

### macOS or Linux

1. Install Python 3.10 or 3.11 from <https://www.python.org/downloads/> and Git from <https://git-scm.com/downloads>.
2. Open Terminal and paste these commands one line at a time:

```bash
git clone https://github.com/fymunotbready-ctrl/multi-task-rl.git
cd multi-task-rl
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

3. Wait for a line containing `Running on local URL`.
4. Open the printed address, normally <http://127.0.0.1:7860>, in a web browser.

### Windows PowerShell

1. Install Python 3.10 or 3.11 from <https://www.python.org/downloads/> and Git from <https://git-scm.com/downloads>.
2. Open PowerShell and paste these commands one line at a time:

```powershell
git clone https://github.com/fymunotbready-ctrl/multi-task-rl.git
cd multi-task-rl
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

3. Wait for a line containing `Running on local URL`.
4. Open the printed address, normally <http://127.0.0.1:7860>, in a web browser.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass`, then repeat `.venv\Scripts\Activate.ps1`.

## What was verified

The Gradio app starts locally, every command produces streamed RGB frames, and
the complete repository test suite passes. The tracked `models/` files total
5,351,476 bytes and the tracked `assets/` files total 1,550,529 bytes:
6,902,005 bytes (6.58 MiB) combined. This is far below the roughly 1 GB
free-Space concern threshold. A live Hugging Face Space was not created or
tested because that requires the owner's Hugging Face account and credentials.
