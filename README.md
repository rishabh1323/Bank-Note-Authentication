# Bank Note Authentication

A python based machine learning application to predict whether the bank note is authentic or a fake, based on user input using supervised machine learning techniques like Logistic Regression, Support Vector Machines and Random Forests.

The application is live [here](http://the-ml-dl-app.herokuapp.com/machine-learning/bank-note-authentication)

## Dataset
- The dataset used is Bank Note Authentication UCI Data
- [Link to Dataset](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

## Technologies Used
![Python](https://img.shields.io/badge/-Python-FFFFFF?style=flat&logo=python&logoColor=3776AB)&nbsp;&nbsp;&nbsp;
![Pandas](https://img.shields.io/badge/-Pandas-FFFFFF?style=flat&logo=pandas&logoColor=150458)&nbsp;&nbsp;&nbsp;
![Numpy](https://img.shields.io/badge/-NumPy-FFFFFF?style=flat&logo=numpy&logoColor=013243)&nbsp;&nbsp;&nbsp;
![Matplotlib](https://img.shields.io/badge/-Matplotlib-FFFFFF?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAEk0lEQVQ4T32VfUgbZxzH73KX95iX80xitMbK5qTWdBoMU7p28a20sCG2f7SwIYylY2+Q1o51yrSuKoOKupWxabZANxgbSPWPdSNCfKmCGOusZhYtlNoaF/N6iUkuL5fcjSerLrO6++fgud/zud/3nu/vezC0/4VUVFScbGpq0ut0uiMymUwBygiCcNlstgcjIyPjCwsLdyEISu3dDu9d0Gg0r/b39195+SX1i8G5UUzdeGkzFoshoE4gECTBnaZpeGpqymE0GnuXl5enMxmZQJbBYPig69pnBvrukFK6dkcIU3GW4/yPm2wxToFNPB4vxWKxIAzD4iwWiyFJEjIajbdMJtPX4D2gZhdoMBg+6ujoeA+TSRPEt81FcmIFRRGEcddf84jKzwT+AXJSoVCQQ1FMeh8AO51Ofmdn5zcmk+nmLhDInJ2d7efxeLDdbpcqROxk1s/v5AkSQdZGQX0Mqr/sAsVhaoMXxU1yZfwTBy4pIldXV8VlZWWBWCzGVFVVXQLywZsQq9V6u6amJs/r9fKysrIon8/HDf85IS6c7sFCAhXN+fD2IwRBmPXwT4owsyTKS17dgGEYys7OjodCITaO47Hx8fHN2traJlir1eptNlsv6MDv93PBQwAGhZxZs0yxNipae2PQJczOjTu5N/IPIe86k6QkJRKJkju1QDrYr9PprsBdXV3X29raTm86ngixbHkcRVF6e3ubLZPJEsv3F7Hc8U48ojkbTBUfjRPMpEQUPR0oKSkJEgTBEYvFFIqiDFAE4D09Pb/BFotluKGhoXD13i1VLvydhEhqEvycUwFYWEGibAG9uXpfzNh/lfgrjlEqSWUAx5UxoVCY5HK5KaAEgNxuN08ul8fGxsbWYZvNNlNaWip6uPhDrkZ8MyuZTMEoijCRGMKKQGXxJP94ZHJyS+qipDAjeoU8UZkTUGKcBI7j8R2gx+Ph5eTkxFZWVsL/CwwxR+N/PFYjWyEckgiwxHH9qS2JRJrwer1A4v7APZLFgaSG4uINwblHhei9NUhQUw6HZGo6RZIhVDzzu0jyWrNHnl9I7ifZYrGsw93d3Z+3traeyTwUv5/gWuZISWkeQR0+rA7bE3YpOMUCJ0Kx7lxXii7ceArL8ilgsecO5SDbhMNhtKCgIEJRFGv4r2E1AJ5TnXvim/kF5099iftOfux7ofb8FviO/7FNprF9Ph8PBACwBIfDoWEYZoB5h1xDxQB4UXHxodfr5kWG2/PljmmuV/t2kFv9pk+uUEStVqujrq7ubHom94yerLi4eJvP5ydBdwC+GFrEQF15Vrkf+DMR2UYTgxeKuIF1tufYW0H89ctPq6urd0cvnT4gHNrb299XqVQkQRBchmHSowW6JEkS3YkvhmFgYOTQ42WhcKoPD9a3u7/4anDAbDb/Gw7P8iwdXwMDA80CgSCdeWAUaZqGotFoGgi6zowvj9uFXv209Xuz2fx8fO2EJJDf19fXotfrD4HMA+uZHe4E7MTExEZLS0vv0tLSzEEBm7mOaLXaE42NjfrKysojGIYpn4XH1vz8/IPR0dGJg34BfwP5MXT+u6N2TgAAAABJRU5ErkJggg==)&nbsp;&nbsp;&nbsp;
![Seaborn](https://img.shields.io/badge/-Seaborn-FFFFFF?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAADcUlEQVQ4T42VfUhTexjHv2ebO2c7Z+A2c/Nq6KytaSSXouRKcQcFXbhcyMjMiprUgrTb6x8RQYlgl/tHEZS7kMW6Fy4qRhElRhAYvvRGVFZmzt6VNq+7x9zZPGdtZ3FOOGdt1vnv8DzP5/e8fH/Pj8Asn15vWpSZmVeq0ehyJbfJyeDI+PjwHZb1P04XRqQwKKzWpVuqqpwHyxxldlOBCSRNym5CSID/tR+9nb0Dzc3n//R67/0DQExmzADSNG1av353y84DLgevJTHOCykTyaRIUGEBfx1v6mxrO1XJcdzolGMCyDBMdm1tQ1fVrg22t8FQ2kZkKBVQEAQisRjmMjSaT7cMNjYeXjEFnQIqqqsP3dhTv9eRCmbUUqAIJThOABeOICbGQWvUYGg18o0MDu071unx/LFSKl8GWq1Lnf9eavX4FNOJqRQKmBgtwlwEg+8C8LGps6YpNX406+DatLna6713XgYeOdL4bJVzjZ2PxWDQkEAUCLAh9L8NICx8nE0Isi1Hz+B1V9dAfX1tESFJo7mto48yZ4HleAyPTWAiHElAtGQG9Awl/0v2dAcs1KlQvXFtCWGxLNl+tq216fbLxKCgIVVYkKNHNCQgMMbhv0BQBs4x6mDMYqCiSTx/z2JSiCYOLi3Mhqui0kUUFzuOnm75u+7WwAjMeho5Og38Iyzu970BnxSQXDdFqrCkJB+mXAPeB8Nyf3+y52LXhq11MvDStQt1z4Z88A75MPhqFPH4N9smOxAEYLNkwzrfjKL5ZpT/sq5OLvmEu6mp/ebA91HSeP36sx37a1wueShnzl3s6340AiIWASGEoSSAuChiPMjDPDcHk1Fggo/NeuDqZfnYsa2yJCEb45wf7Nev3YVez0CpVMrBPt//MJsNICk1xDiBPPs8sBMCQvxMKRXkGRAZffFZNlPCPnnylMftvpwSKPmMjX2QbYYsA0haC5HUQJqZEIlizcpi7K1xTQsbgHz1SkvLHN3dfV9lmAxMzr6ouADLV5Sg/UpHp8fTMH31pABpOdTUNHRZCi2227f6Z5ScDuh0rsaDB08G3e6vl4MMkKAVFb+3lpf/5nj4cAg9PU/kHn4JzM3LwuLFNrRf7Ui/vpJGmFiwNpvNrlZnIB6PIxTiodNpIYoinj7t/74F+6UuUj8B/jssO5z2CfgErpV/0gFRZ9wAAAAASUVORK5CYII=)&nbsp;&nbsp;&nbsp;
![Scikit Learn](https://img.shields.io/badge/-ScikitLearn-FFFFFF?style=flat&logo=scikitlearn&logoColor=F7931E)&nbsp;&nbsp;&nbsp;
![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-FFFFFF?style=flat&logo=jupyter&logoColor=F37626)

## Installations Required
For running the source code on your local machine, the following dependencies are required.
- Python
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn

## Launch
For local installation, follow these steps:
1. Download source code from this repository or click [here](https://github.com/rishabh1323/Bank-Note-Authentication/archive/refs/heads/main.zip).
2. Extract files to desired directory.
3. [Download](https://www.python.org/downloads/) and install Python3 if not done already.
4. Create a new python virtual environment.
```
python3 -m venv tutorial-env
```
5. Once you’ve created a virtual environment, you may activate it.  

`On Windows, run:`
```
tutorial-env\Scripts\activate.bat
```
`On Unix or MacOS, run:`
```
source tutorial-env/bin/activate
```
> Refer to [python documentation](https://docs.python.org/3/tutorial/venv.html) for more information on virtual environments.  
6. Install the required dependecies.
```
pip install numpy pandas seaborn matplotlib scikit-learn
```
7. Launch the Jupyter Notebooks or run the python files now.