







































#Q1
import math

def letter_to_number(char):
    if 'A' <= char <= 'Z':
        return ord(char) - ord('A')
    elif char == ' ':
        return 26
    return None # Handle other characters if necessary

def number_to_letter(num):
    if 0 <= num <= 25:
        return chr(num + ord('A'))
    elif num == 26:
        return ' '
    return None # Handle other numbers if necessary

def rsa_encrypt(message, e, n, block_size):
    encoded_numbers = []
    for char in message.upper():
        num = letter_to_number(char)
        if num is not None:
            encoded_numbers.append(num)

    # Group numbers into blocks based on block_size
    blocks = []
    current_block = 0
    # Corrected block grouping logic
    for i, num in enumerate(encoded_numbers):
        if block_size == 1:
            blocks.append(num)
        else:
             # This logic for block_size > 1 was a previous attempt and might not be suitable for the 0-26 encoding scheme.
             # Given n=100, block_size must be 1. The following block is likely not needed for this specific problem but kept for generality.
             current_block = current_block * (10**(2*block_size)) + num # Adjust base based on block size
             if (i + 1) % block_size == 0 or i == len(encoded_numbers) - 1:
                blocks.append(current_block)
                current_block = 0


    # Encrypt each block
    encrypted_blocks = []
    for block in blocks:
        if block >= n:
            print(f"Warning: Block value {block} is greater than or equal to n ({n}). This may cause issues.")
        encrypted_block = pow(block, e, n)
        encrypted_blocks.append(encrypted_block)

    return encrypted_blocks

# Given parameters
e = 13
n = 100
message = "HOW ARE YOU"

# Determine block size such that P < n.
# Since n=100, a block size of 2 letters (00-25, 26) will result in a maximum block value of 2626, which is > 100.
# A block size of 1 letter (0-26) will result in a maximum block value of 26, which is < 100.
# So, we should use a block size of 1 letter.
block_size = 1

encrypted_message = rsa_encrypt(message, e, n, block_size)
print(f"Original message: {message}")
print(f"Public Key (e, n): ({e}, {n})")
print(f"Encrypted message (blocks): {encrypted_message}")

# To display the encrypted message as pairs of numbers (as often shown in examples)
print("Encrypted message (pairs of numbers):")
print(' '.join(map(str, encrypted_message)))

#Q2
import math

def find_prime_factors(n):
    factors = set()
    # Check for factor 2
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    # Check for odd factors from 3 up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.add(i)
            n //= i
    # If n is still greater than 2, it must be a prime factor
    if n > 2:
        factors.add(n)
    return sorted(list(factors))

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        gcd, x, y = extended_gcd(b, a % b)
        return gcd, y, x - (a // b) * y

def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None # Modular inverse does not exist
    else:
        return x % m

# Given parameters
e = 17
n = 187

# Step 1: Find prime factors of n
prime_factors = find_prime_factors(n)

if len(prime_factors) == 2:
    p, q = prime_factors
    print(f"Prime factors of n={n} are p={p} and q={q}")

    # Step 2: Calculate phi(n)
    phi_n = (p - 1) * (q - 1)
    print(f"phi(n) = (p-1)(q-1) = ({p}-1)({q}-1) = {phi_n}")

    # Step 3: Find d, the modular inverse of e mod phi(n)
    d = mod_inverse(e, phi_n)

    if d is not None:
        print(f"The private key d is: {d}")
        # Verification: (e * d) mod phi(n) should be 1
        verification = (e * d) % phi_n
        print(f"Verification: (e * d) mod phi(n) = ({e} * {d}) mod {phi_n} = {verification}")
    else:
        print(f"Modular inverse of e={e} mod phi(n)={phi_n} does not exist.")

else:
    print(f"Could not find two distinct prime factors for n={n}. RSA requires n to be a product of two distinct primes.")

e = 3
n = 35
ciphertext = 22

print(f"Public Key (e, n): ({e}, {n})")
print(f"Intercepted Ciphertext C: {ciphertext}")

# Store the sequence of encrypted values
encrypted_sequence = [ciphertext]
current_value = ciphertext

print("\nPerforming cyclic attack:")
while True:
    # Encrypt the current value using the public key
    next_value = pow(current_value, e, n)
    print(f"Encrypting {current_value} -> {next_value} (mod {n})")

    # Check if we have cycled back to the original ciphertext
    if next_value == ciphertext:
        print(f"\nCycled back to the original ciphertext: {next_value}")
        # The plaintext is the value before the original ciphertext was reached again
        # If the sequence has only one element (the original ciphertext), it means
        # the first encryption already resulted in the original ciphertext. This is unlikely
        # for a meaningful plaintext, but the logic should handle it.
        if len(encrypted_sequence) > 0:
             plaintext = encrypted_sequence[-1]
             print(f"The plaintext is the value before the cycle completed: {plaintext}")
        else:
             # This case should ideally not happen with the logic above, but as a fallback
             print("Could not determine plaintext from the sequence.")
        break
    # Check if we've seen this value before (indicates a shorter cycle that doesn't include the original ciphertext initially)
    # While the problem description implies cycling back to C, in some cases a cycle might form before C.
    # However, the most straightforward interpretation for this problem is cycling back to the initial C.
    # We'll stick to the problem's description of cycling back to C.
    elif next_value in encrypted_sequence:
         print(f"\nDetected a cycle before returning to the original ciphertext {ciphertext}. This might not be the intended scenario for this problem.")
         print(f"Value {next_value} was already seen.")
         break

    encrypted_sequence.append(next_value)
    current_value = next_value

# To verify, we can decrypt the plaintext with the private key if we had it.
# For demonstration purposes, we'll just show the cycling process.

#Q4
# Chosen Ciphertext Attack on RSA

import random
from math import gcd

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        gcd_val, x, y = extended_gcd(b, a % b)
        return gcd_val, y, x - (a // b) * y

def mod_inverse(a, m):
    gcd_val, x, y = extended_gcd(a, m)
    if gcd_val != 1:
        return None # Modular inverse does not exist
    else:
        return x % m

# Bob's RSA public key (known to Eve)
e = 7
n = 143

# Alice's plaintext and the resulting ciphertext (intercepted by Eve)
# Eve knows C, but does NOT know P initially. We include P here for verification.
P = 8
C = 57

print(f"Bob's Public Key (e, n): ({e}, {n})")
print(f"Intercepted Ciphertext C: {C}")
print(f"(For verification, the original plaintext P is: {P})")

# --- Eve's Steps ---

# a. Eve chooses a random integer X in Zn* (coprime to n)
# Zn* is the set of integers from 1 to n-1 that are coprime to n.
# We'll choose a random X and check if it's coprime to n.
X = 0
while gcd(X, n) != 1 or X == 0:
    X = random.randint(1, n - 1)

print(f"\nEve chooses a random integer X: {X}")

# b. Eve calculates Y = C × X^e mod n
Y = (C * pow(X, e, n)) % n
print(f"Eve calculates Y = C * X^e mod n = ({C} * {X}^{e} mod {n}) mod {n} = {Y}")

# c. Eve sends Y to Bob for decryption and gets Z = Y^d mod n
# We need Bob's private key d to simulate Bob's decryption.
# First, find p and q for n=143. 143 = 11 * 13.
p = 11
q = 13
phi_n = (p - 1) * (q - 1) # (11-1)*(13-1) = 10 * 12 = 120
# Find d such that e * d = 1 mod phi_n (7 * d = 1 mod 120)
d = mod_inverse(e, phi_n) # mod_inverse(7, 120) should be 103 (7 * 103 = 721 = 6 * 120 + 1)
print(f"(Simulating Bob's private key d = {d})")

# Bob decrypts Y to get Z
Z = pow(Y, d, n)
print(f"Bob decrypts Y: Z = Y^d mod n = {Y}^{d} mod {n} = {Z}")

# d. Eve easily finds P because Z = P * X mod n
# Eve needs to calculate P = Z * X^-1 mod n
X_inverse = mod_inverse(X, n)
if X_inverse is None:
    print(f"Error: Modular inverse of X={X} mod n={n} does not exist.")
else:
    found_P = (Z * X_inverse) % n
    print(f"Eve calculates P = Z * X^-1 mod n = ({Z} * {X_inverse}) mod {n} = {found_P}")

    # Verify if the found plaintext matches the original plaintext
    if found_P == P:
        print("\nSuccess! Eve found the correct plaintext.")
    else:
        print("\nAttack failed. Found plaintext does not match original.")