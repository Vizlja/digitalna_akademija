import cv2
from PIL import Image

kaskada = "haarcascade_frontalface_default.xml"

kaskadaLice = cv2.CascadeClassifier(kaskada)

slika = cv2.imread("osoba1.osam.jpg")
crnoBijelo = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

lica = kaskadaLice.detectMultiScale(
    crnoBijelo,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Broj pronađenih lica: ", len(lica))

print(lica) 

for (x, y, w, h) in lica:
    izrezana = crnoBijelo[y:y+h, x:x+w]
    mala = cv2.resize(izrezana,(300,300))
    cv2.imshow("izrezana", mala)
    cv2.imwrite( "izrezana.jpg", mala)
    cv2.waitKey(0)
