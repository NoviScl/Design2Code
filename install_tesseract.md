Pytesseract requires the [tesseract](https://tesseract-ocr.github.io/tessdoc/Compiling.html) package, which in turn requires the [leptonica](http://www.leptonica.org/) package. 

In case you don't have root access, you will have to install these packages from source. You can follow the steps below.

1. Download the source package from [this link](http://www.leptonica.org/download.html) for Leptonica. 
2. Run ```./configure --prefix=$HOME``` (change ```$HOME``` to the directory where you want to install the package).
3. Run ```make``` and then ```make install```.

4. Download tesseract source by running ```git clone https://github.com/tesseract-ocr/tesseract.git```.
5. Follow the steps listed [here](https://tesseract-ocr.github.io/tessdoc/Compiling-%E2%80%93-GitInstallation.md) to install from source. Remember to use the ```--prefix``` flag while running ```./configure```.
6. Follow the steps about downloading language data and put in where you stored tesseract (e.g., ```$HOME/share/tessdata```).