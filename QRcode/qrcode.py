# Import QRCode from pyqrcode
import pyqrcode
import png
from pyqrcode import QRCode

# String which represents the QR code
s = "https://webhook.etechats.com.br/webhook/ufg_tela"

# Generate QR code
url = pyqrcode.create(s)

# Create and save the svg file naming "myqr.svg"
url.svg("myqr.svg", scale=8)

# Create and save the png file naming "myqr.png"
qr =url.png("myqr.png", scale=6)
print(url, qr) 