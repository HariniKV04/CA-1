











































HOST = "127.0.0.1"
PORT = 50007
A_KEY = 5
B_KEY = 8

def start_server(host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            while True:
                data = conn.recv(4096)
                if not data:
                    print("Connection closed by client")
                    break
                ciphertext = data.decode('utf-8')
                print("Received ciphertext:", ciphertext)
                try:
                    plaintext = decrypt(ciphertext, A_KEY, B_KEY)
                except Exception as e:
                    plaintext = f"[decryption error: {e}]"
                print("Decrypted plaintext:", plaintext)
                # send ack
                ack = f"Server received and decrypted message: {plaintext}"
                conn.sendall(ack.encode('utf-8'))


start_server()

"""Client.py"""

HOST = "127.0.0.1"
PORT = 50007
A_KEY = 5
B_KEY = 8

def send_message(msg, host=HOST, port=PORT):
    ct = encrypt(msg, A_KEY, B_KEY)
    print("Sending ciphertext:", ct)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(ct.encode('utf-8'))
        data = s.recv(4096)
    print("Received ack:", data.decode('utf-8'))


message = "Hello server, this is a test!"
send_message(message)

"""## Affine Cipher"""

from math import gcd
import socket

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
M = len(ALPHABET)

def modinv(a, m):
    a = a % m
    if gcd(a, m) != 1:
        raise ValueError(f"{a} has no modular inverse mod {m}")
    t0, t1 = 0, 1
    r0, r1 = m, a
    while r1 != 0:
        q = r0 // r1
        r0, r1, t0, t1 = r1, r0 - q * r1, t1, t0 - q * t1
    inv = t0 % m
    return inv

def encrypt(plaintext, a, b):
    if gcd(a, M) != 1:
        raise ValueError(f"a={a} is invalid since gcd({a},{M}) != 1")
    plaintext = plaintext.upper()
    out = []
    for ch in plaintext:
        if 'A' <= ch <= 'Z':
            x = ord(ch) - ord('A')
            y = (a * x + b) % M
            out.append(chr(y + ord('A')))
        else:
            out.append(ch)
    return ''.join(out)

def decrypt(ciphertext, a, b):
    if gcd(a, M) != 1:
        raise ValueError(f"a={a} is invalid since gcd({a},{M}) != 1")
    a_inv = modinv(a, M)
    ciphertext = ciphertext.upper()
    out = []
    for ch in ciphertext:
        if 'A' <= ch <= 'Z':
            y = ord(ch) - ord('A')
            x = (a_inv * (y - b)) % M
            out.append(chr(x + ord('A')))
        else:
            out.append(ch)
    return ''.join(out)

a, b = 5, 8
pt = "Hello, Affine!"
ct = encrypt(pt, a, b)
dec = decrypt(ct, a, b)
print("Plain:", pt)
print("Cipher:", ct)
print("Decrypted:", dec)

"""## Caesar Cipher"""

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
M = len(ALPHABET)

def sanitize_text(text):
    return text.upper()

def encrypt(plaintext, shift):
    plaintext = sanitize_text(plaintext)
    out = []
    for ch in plaintext:
        if 'A' <= ch <= 'Z':
            x = ord(ch) - ord('A')
            y = (x + shift) % M
            out.append(chr(y + ord('A')))
        else:
            out.append(ch)
    return ''.join(out)

def decrypt(ciphertext, shift):
    ciphertext = sanitize_text(ciphertext)
    out = []
    for ch in ciphertext:
        if 'A' <= ch <= 'Z':
            y = ord(ch) - ord('A')
            x = (y - shift) % M
            out.append(chr(x + ord('A')))
        else:
            out.append(ch)
    return ''.join(out)

shift = 3
pt = "Hello Caesar!"
ct = encrypt(pt, shift)
dec = decrypt(ct, shift)
print("Plain:", pt)
print("Cipher:", ct)
print("Decrypted:", dec)

"""## RSA"""

import socket
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

HOST, PORT = "127.0.0.1", 50007

# Generate keypair (do this once, or load from file)
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
public_key = private_key.public_key()

# Save public key so client can use it
with open("server_public.pem", "wb") as f:
    f.write(public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ))

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            data = conn.recv(4096)
            if data:
                plaintext = private_key.decrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                print("Decrypted:", plaintext.decode())
                conn.sendall(b"Message received and decrypted!")

if __name__ == "__main__":
    start_server()

import socket
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

HOST, PORT = "127.0.0.1", 50007

# Load the server's public key (must match!)
with open("server_public.pem", "rb") as f:
    server_public_key = serialization.load_pem_public_key(f.read())

def send_message(msg):
    ciphertext = server_public_key.encrypt(
        msg.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(ciphertext)
        ack = s.recv(4096)
        print("Server reply:", ack.decode())

if __name__ == "__main__":
    send_message("Hello secure server with RSA!")

import random
from math import gcd

def is_prime(n, k=5):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bit_length=8):
    while True:
        p = random.getrandbits(bit_length)
        if is_prime(p):
            return p

def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('No modular inverse')
    return x % m

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

def generate_keys(bit_length=8):
    p = generate_prime(bit_length)
    q = generate_prime(bit_length)
    while q == p:
        q = generate_prime(bit_length)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537
    if gcd(e, phi) != 1:
        e = 3
        while gcd(e, phi) != 1:
            e += 2

    d = modinv(e, phi)
    return ((e, n), (d, n))

# RSA encryption
def encrypt(message, pub_key):
    e, n = pub_key
    message_int = [pow(ord(c), e, n) for c in message]
    return message_int

# RSA decryption
def decrypt(cipher, priv_key):
    d, n = priv_key
    chars = [chr(pow(c, d, n)) for c in cipher]
    return ''.join(chars)


public, private = generate_keys(bit_length=8)
print("Public key:", public)
print("Private key:", private)

msg = "HELLO"
cipher = encrypt(msg, public)
print("Cipher:", cipher)

plain = decrypt(cipher, private)
print("Decrypted:", plain)

"""## AES"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

key = AESGCM.generate_key(bit_length=256)
with open("aes_key.bin", "wb") as f:
    f.write(key)
print("AES key saved to aes_key.bin")

import socket
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

HOST, PORT = "127.0.0.1", 50007

# Load the same key used by client
with open("aes_key.bin", "rb") as f:
    key = f.read()

aesgcm = AESGCM(key)

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            data = conn.recv(4096)
            if data:
                nonce, ciphertext = data[:12], data[12:]
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                print("Decrypted:", plaintext.decode())
                conn.sendall(b"Message received and decrypted!")

if __name__ == "__main__":
    start_server()

import socket, os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

HOST, PORT = "127.0.0.1", 50007

# Load the same key as server
with open("aes_key.bin", "rb") as f:
    key = f.read()

aesgcm = AESGCM(key)

def send_message(msg):
    nonce = os.urandom(12)  # unique per message
    ciphertext = aesgcm.encrypt(nonce, msg.encode(), None)
    payload = nonce + ciphertext
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(payload)
        ack = s.recv(4096)
        print("Server reply:", ack.decode())

if __name__ == "__main__":
    send_message("Hello secure AES world!")

import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def generate_key():
  return os.urandom(32)

def encrypt_aes_gcm(key, plaintext):
  nonce = os.urandom(12)
  cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
  encryptor = cipher.encryptor()
  ciphertext = encryptor.update(plaintext) + encryptor.finalize()
  tag = encryptor.tag
  return nonce, ciphertext, tag

def decrypt_aes_gcm(key, nonce, ciphertext, tag):
  try:
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return decrypted_plaintext
  except Exception as e:
    print(f"Decryption failed: {e}")
    return None

secret_key = generate_key()
print(f"Generated Key: {secret_key.hex()}")

original_plaintext = b"This is a secret message that will be encrypted."
print(f"\nOriginal Plaintext: {original_plaintext.decode()}")

nonce, ciphertext, tag = encrypt_aes_gcm(secret_key, original_plaintext)
print(f"\nNonce: {nonce.hex()}")
print(f"Ciphertext: {ciphertext.hex()}")
print(f"Authentication Tag: {tag.hex()}")

decrypted_plaintext = decrypt_aes_gcm(secret_key, nonce, ciphertext, tag)

if decrypted_plaintext:
  print(f"\nDecrypted Plaintext: {decrypted_plaintext.decode()}")
  assert original_plaintext == decrypted_plaintext

import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def bytes_to_hex_str(data):
    if not data:
        return ""
    return ' '.join(f'{b:02x}' for b in data)

def investigate_mode(mode_name, key, plaintext):
    print(f"opmode:\n\n{mode_name}:\n")
    print(f"input : {bytes_to_hex_str(plaintext)}")

    iv = b'\x00' * 16 if mode_name in ['CBC', 'CFB'] else None

    if mode_name == 'ECB':
        mode = modes.ECB()
    elif mode_name == 'CBC':
        mode = modes.CBC(iv)
    elif mode_name == 'CFB':
        mode = modes.CFB(iv)
    else:
        raise ValueError("Unsupported mode specified")

    cipher_instance = Cipher(algorithms.AES(key), mode, backend=default_backend())

    encryptor = cipher_instance.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    print(f"cipher: {bytes_to_hex_str(ciphertext)}")

    cipher_list = bytearray(ciphertext)

    modification_index = 20
    original_byte = cipher_list[modification_index]
    modified_byte = (original_byte - 1) & 0xFF
    cipher_list[modification_index] = modified_byte
    modified_ciphertext = bytes(cipher_list)

    print(f"\nModifying random byte: {original_byte:02x}->{modified_byte:02x}\n")

    decryptor_cipher = Cipher(algorithms.AES(key), mode, backend=default_backend())
    decryptor = decryptor_cipher.decryptor()
    corrupted_plaintext = decryptor.update(modified_ciphertext) + decryptor.finalize()

    print(f"plain : {bytes_to_hex_str(corrupted_plaintext)}")

secret_key = b'R\xd8\x1e\xfd\xbb\xe5\x87\xe2\x93\x1e\x8e\xbd\x92\xe0\x99\xd8\xf6\x95\x88\x99T[\xf6\xf8\x91\xe7\x04\x957\x1a\xf5\xc4'
patterned_plaintext = bytes(range(16)) * 2
modes_to_investigate = ['ECB', 'CBC', 'CFB']

for mode in modes_to_investigate:
    investigate_mode(mode, secret_key, patterned_plaintext)