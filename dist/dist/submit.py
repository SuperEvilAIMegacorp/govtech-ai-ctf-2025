import argparse, base64, hashlib, io, requests
from PIL import Image

def get_challenge(s):
    return requests.get(s.rstrip('/') + '/challenge', timeout=30).json()

def solve_pow(chal, diff):
    print("solving proof of work...")
    need = "0" * int(diff); n = 0
    while True:
        nonce = str(n)
        if hashlib.sha256((nonce + chal).encode()).hexdigest().startswith(need):
            return nonce
        n += 1

def image_b64(path):
    buf = io.BytesIO()
    Image.open(path).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def post_infer(s, cid, nonce, img):
    return requests.post(s.rstrip('/') + '/infer',
        json={"challenge_id": cid, "nonce": nonce, "image": img}, timeout=60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--image", required=True)
    a = ap.parse_args()

    ch = get_challenge(a.server)
    nonce = solve_pow(ch["challenge"], ch["difficulty"])
    resp = post_infer(a.server, ch["challenge_id"], nonce, image_b64(a.image))

    try: print(resp.json())
    except Exception: print(resp.text)

if __name__ == "__main__":
    main()
