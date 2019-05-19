from flask import render_template, url_for, redirect, session, request
from expansion import app
from expansion.forms import QueryForm, IndexForm

import expansion.functions as funct
from collections import OrderedDict
from operator import itemgetter
import os

# to convert string to dict
import ast
import warnings
warnings.filterwarnings("ignore")

indx = open("../QueryExpansion/Index.txt", 'r')
indx = indx.read()


# --------------------------------------------------------------------------------------------

proximityIndex  = {"index":ast.literal_eval(indx)}
document = {"name": "", "content": "", "number":""}
preprocess = {"content":""}
searchRes = {"number": ""}
terms = {"all" : ""}

document["name"] = funct.all_filename()
document["content"] = funct.all_content()
document["number"] = funct.generateDocNumber(document["name"])
preprocess["content"] = funct.preprocess(document["content"])
terms["all"] = funct.getAllTerms(preprocess["content"])


# HOME PAGE
@app.route("/")
@app.route("/search", methods=['GET', 'POST'])
def search():
	form = QueryForm()

	if form.validate_on_submit():
		query = (form.query.data)
		return redirect(url_for('result', query=query))
	return render_template('search.html', form = form )

# RESULT PAGE
@app.route("/result/<string:query>", methods=['GET', 'POST'])
def result(query):
	form = QueryForm()

	search_score = funct.search(query,proximityIndex["index"])
	name = document["name"]
	content = document["content"]
	searchRes["number"] = list(search_score.keys())
	sscore = search_score
	# print(sscore)
	search_score = list(search_score.keys())

	# print(sscore)
	# print(search_score)

	if(len(search_score)<1):
		message = "No result found"
	else:
		message = ""

	if form.validate_on_submit():
		query = (form.query.data)
		return redirect(url_for('result', query=query))

	userQuery = query
	return render_template('result.html', query = userQuery, form=form, score=search_score, sscore = sscore, name=name, content=content, message=message)

# INDEXING PAGE
@app.route("/index", methods=['GET', 'POST'])
def index():
	form = IndexForm()

	res = proximityIndex["index"] 

	if form.validate_on_submit():
		new_index = funct.generateIndex()
		document["name"] = funct.all_filename()
		document["content"] = funct.all_content()
		document["number"] = funct.generateDocNumber(document["name"])
		preprocess["content"] = funct.preprocess(document["content"])
		a = funct.saveIndex(new_index)
		proximityIndex["index"] = new_index
		return redirect(url_for('index'))

	return render_template('index.html', form=form, res = res)


# EXPAND PAGE
@app.route("/expand/<string:query>")
def expand(query):

	queryVec = funct.vector(query, terms["all"])

	relVec = []
	total = []
	totalDict = {}
	
	# res = funct.relevance(searchRes["number"], document["content"])
	res = funct.relevance(searchRes["number"], preprocess["content"])

	
	for words in res["rel"]:
		relVec.append(funct.vector(words, terms["all"]))
		

	relVecSum = funct.sumVector(relVec)

	mulQuery = funct.multiplyVector(0.8, queryVec)
	mulRelevan = funct.multiplyVector(0.5, relVecSum)

	for i in range(len(mulQuery)):
		total.append(mulQuery[i] + mulRelevan[i])
		
	for i in range(len(mulQuery)):
		if(total[i] > 0):
			totalDict[terms["all"][i]] = total[i]
	
	# new = sorted(totalDict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
	new = OrderedDict(sorted(totalDict.items(), key=itemgetter(1), reverse=True))

	new_query = {}

	for i in range(5):
		new_query[str(list(new.keys())[i])] = list(new.values())[i]
	
	str_query = []
	for i in range(len(new_query)):
		str_query.append(list(new_query.keys())[i])

	str_query = " ".join(str_query)
	
	return render_template('expand.html', output=str_query)

# HELP PAGE
@app.route("/pdfView/<int:id_doc>")
def pdfView(id_doc):
	filename = list(document["name"])[int(id_doc)]
	# filename = "../QueryExpansion/expansion/collection/"+filename
	return render_template('pdfView.html', filename=filename)

# ABOUT PAGE
@app.route("/about")
def about():
	
	return render_template('about.html', output=preprocess["content"])



# =========================================================================================
# TEST PAGE
@app.route("/test")
def test():
	output = document
	# proximityIndex["index"] = output
	return render_template('test.html', output = output)