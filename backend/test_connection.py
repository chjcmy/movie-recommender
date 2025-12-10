import socket
import os

host = "192.168.55.121"
port = 5433

print(f"Testing connection to {host}:{port}...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
try:
    s.connect((host, port))
    print("Connection successful!")
    s.close()
except Exception as e:
    print(f"Connection failed: {e}")
