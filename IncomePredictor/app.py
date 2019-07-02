from flask import Flask,render_template,request
import os
import pickle
import numpy as np

#prediction funtion

def ValuePredictor(to_ptedict_list):
	to_predict=np.array(to_ptedict_list).reshape(1,12)
	load_model=pickle.load(open("model.pkl","rb"))
	result=load_model.predict(to_predict)
	return result[0]


app=Flask('__name__')

@app.route('/')
@app.route('/index')
def show_predict_stock_form():
	return render_template('prediction.html')

@app.route('/result',methods=['POST'])
def results():
	form=request.form
	if request.method == 'POST':
		#write your function that load the model we use pickle also
		to_predict_list=request.form.to_dict()
		to_predict_list=list(to_predict_list.values())
		to_predict_list=list(map(float,to_predict_list))
		result=ValuePredictor(to_predict_list)

		if int(result)==1:
			prediction='Income more then 50k'
		else:
			prediction='Income less then 50k'	
		return render_template('result.html',prediction=prediction)

if __name__ =='__main__':
	app.run("localhost","9999",debug=True)