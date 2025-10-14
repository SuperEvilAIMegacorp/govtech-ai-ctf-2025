# StrideSafe
## Problem Description
Singapore's lamp posts are getting smarter. They don't just light the way, they watch over the pavements.
Your next-gen chip has been selected for testing. Can your chip distinguish pedestrians from bicycles and PMDs (personal mobility devices)?
Pass the test, and your chip will earn deployment on Singapore's smart lamp posts. Fail, and hazards roam free on pedestrian walkways.

## Thoughts
Difficulty: ★★☆☆☆ (2/5 stars)

Time taken: ~30 minutes

AI Assistance: 🧠🧠🧠🧠🧠 

The challenge name "flag the hazards" was a clever hint at the classification problem. Working with CLIP models and generating QR codes from classification results was a really creative challenge design!

## Initial Reconnaissance
<p align="center">
<img width="70%" src="https://github.com/user-attachments/assets/2997d1d8-8aaf-4335-bdfe-2b93090ed284" alt="StrideSafe website" />
</p>
<p align="center"><em>First look at the StrideSafe website</em></p>

## Discovering the Hidden Clue
Similar to the "Don't Chao Keng" challenge, inspecting the page source revealed a helpful comment:
<p align="center">
<img width="70%" src="https://github.com/user-attachments/assets/6233a062-2f1a-41da-a232-79a49765cf0d" alt="HTML comment discovery" />
</p>
<p align="center"><em>HTML comment revealed in inspect element</em></p>

## Building the Solution
Using the initial `deploy-script.py` provided together with a rough implementation of CLIP classification, the first picture generated made it fairly obvious that it was supposed to be a QR code(although it was very fuzzy and the layering was weird). I wasn't sure if this was due to my model choice since I didn't know which OpenAI CLIP model to use so I ended up making my own script (seen in `stridesafe.py`) to test multiple models... The output QR codes are as below:

<img width="2958" height="1021" alt="image" src="https://github.com/user-attachments/assets/2ac0be45-51b1-4d81-8e3b-438ca97483dc" />

<p align="center">
<img width="20%" src="https://github.com/user-attachments/assets/4a32740b-20ad-4b2d-ba6d-af71b2b1dfe3" alt="QR code scan result" />
</p>
<p align="center"><em>Scanning the QR code reveals the flag</em></p>

---

<p align="center">
<img src="https://github.com/user-attachments/assets/d9c1ac19-871a-4966-bc72-e6aa2ef6b709" width="60%" alt="jay-renshaw-chit" />
</p>
