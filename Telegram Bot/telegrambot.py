from dotenv import load_dotenv
import os
import telebot
import joblib
import pickle
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

cv = CountVectorizer(max_features=1500)
data = pd.read_csv("TamilFakeAndRealNew.csv")
print(data.head())
corpus = data['clean_text'].fillna(' ')
x = cv.fit_transform(corpus).toarray()

models = ['Alpha Engine', 'Beta Engine', 'Gamma Engine', 'Lambda Engine', 'Sigma Engine', 'Omega Engine']
loadedModels = {}
for i in models:
    with open("./models/"+i+".pkl", 'rb') as selectedModel:
        loadedModels[i] = pickle.load(selectedModel)

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
print(BOT_TOKEN)
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    print(message)
    bot.reply_to(message, "Howdy, how are you doing?")

@bot.message_handler(commands=['detect', 'find', 'truth', 'news'])
def check_accuracy(message):
    # bot.send_chat_action(message.chat.id, action="Typing......")
    news = [GoogleTranslator(source='tamil', target='en').translate(message.text.split('/detect ')[-1]).lower()]
    print(news)
    ans = []
    summary = {}
    summaryText = ''
    count = 0
    for i in list(loadedModels.keys()):
        try:
            predicted = loadedModels[i].predict(cv.transform(news).toarray())[0]
            ans.append(predicted)
            summary[i] = predicted
            if predicted == 0:
                summaryText += i + ':\t\t' + "Fake\n"
            else:
                summaryText += i + ':\t\t' + "Real\n"
            print(i, ans)
            count += 1
        except:
            pass
    print(summary)
    bot.reply_to(message, "The News is " + str((sum(ans)/count) * 100) +"% Real\n\nSummary\n\n"+summaryText)

bot.infinity_polling()