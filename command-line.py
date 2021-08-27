import pickle

lyrics=input('Please enter lyrics:')
if lyrics == '':
    print('No lyrics were given. Have a nice day!')
    quit()


tf=pickle.load(open('tf.pkl','rb'))
rf=pickle.load(open('rf_clf.pkl','rb'))

test=tf.transform([lyrics])
#print('The lyrics you entered belong to',*rf.predict(test))

for i in range(len(rf.classes_)):
    print (round(100*(rf.predict_proba(test)[0][i]),2),
             '% probability that the lyrics you entered belong to',rf.classes_[i])
