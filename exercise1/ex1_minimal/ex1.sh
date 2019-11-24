source ${HOME}/.virtualenvs/pyScience-wEcEHdig/bin/activate

python ex1.py --alpha 0.3 --file ../ex1data1.txt
python ex1.py --alpha 0.1 --file ../ex1data1.txt
python ex1.py --alpha 0.01 --file ../ex1data1.txt
python ex1.py --alpha 0.03 --file ../ex1data1.txt

python ex1.py --alpha 0.3 --file ../ex1data2.txt
python ex1.py --alpha 0.1 --file ../ex1data2.txt
python ex1.py --alpha 0.01 --file ../ex1data2.txt
python ex1.py --alpha 0.03 --file ../ex1data2.txt

deactivate