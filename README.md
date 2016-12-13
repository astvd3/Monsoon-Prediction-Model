# Monsoon-Prediction-Model
A supervised Learning approach to find the trends on which the monsoon rainfall in India Peninsula depends on. The future monsoon predictions can be done using this trends.
### Install

This project requires **Python 2.7** with the following library installed:
- [pandas](http://pandas.pydata.org/)
- [numpy](http://www.numpy.org/)
- [sklearn](http://scikit-learn.org/stable/install.html)
- [Tkinter](https://docs.python.org/2/library/tkinter.html)

### Data

The datasets can be found in the same directory:
`input.csv`
`output.csv`
`onset_ip.csv`
`onset_op.csv`
`test_op.csv`


Note: [Pickle files](https://docs.python.org/2/library/pickle.html) are also present in this folder, which holds the 'best' classifier which was obtained in the Testing phase. This pickle file is further loaded into the app.py file where it is used to create a GUI based application.

### Run

For obtaining the classifier, in a terminal or command window, run the following command:

`python data_process.py`

This will run the `data_process.py` file and execute the machine learning code.

For obtaining the results and plotting it, run the following command

`python test.py`

For obtaining the failed results about onset and plotting it, run the following command

`python onset.py`

For opening the GUI, run the following command

`python app.py`

This will open the GUI where you can enter the attributes about your data or load a csv file and it will show the future prediction for monsoon.
