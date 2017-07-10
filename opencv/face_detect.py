# Face detection using OpenCV
import argparse
import cv2
import os

def main(args):
    src = args.src

    # 画像読み込み
    image = cv2.imread(src)
    # グレイスケール (した方が、処理速度が早いとのこと)
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscaled = cv2.equalizeHist(grayscaled)
    cascade = cv2.CascadeClassifier('./lbpcascade_animeface/lbpcascade_animeface.xml')
    facerect = cascade.detectMultiScale(grayscaled, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
    if len(facerect) != 0:
        print('Detected ' + str(len(facerect)) + ' faces')
        color = (255, 255, 255)
        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
        filepath, extension = os.path.splitext(src)
        dist = filepath + '_detected' + extension
        cv2.imwrite(dist, image)
    else:
        print('No detect')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    main(parser.parse_args())

