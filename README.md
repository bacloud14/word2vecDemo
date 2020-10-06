# word2vecDemo

This is a demo application that is using the amazing DeepLearning library `deeplearning4j` on Android. Please read more about deployment of `dl4j` on Android, and why it is tricky and probably would not make it anyway !

The Python solution here inspired me https://www.kaggle.com/aamaia/emojis-glove so thanks to **amaia**

It simply tries to learn (word embedding) from a small sentences corpus, and predict similarity between emoji keywords and any input sentence in English.

There are two challenges:

- Exclude unused dependencies as possible.
- Revise the model so it can be more accurate and performant (as any ML algorithm).

# Dependencies and resources

Of course the trained model would be ported to this demo app, and cannot be directly trained on a phone device, check this gist used to generate the model using `wikisent2.txt` from https://www.kaggle.com/mikeortman/wikipedia-sentences and `emojis.json` from https://www.kaggle.com/aamaia/emojis-glove/#data

glove data sets can be huge and very much for this learning goal, if you want to train consider to split and test again for accuracy check 

`cat wikisent2.txt | awk 'BEGIN{srand();}{print rand()"\t"$0}' | sort -k1 -n | cut -f2- > myfile.shuffled`

`split -l 1000000 myfile.shuffled`

Check the gist that generates Word2Vec model https://gist.github.com/bacloud14/5b4a6f8a730261d6d7fe6e77d5065139
or https://github.com/bacloud14/text2emoji_backend



# Contribution

Please see open issues for a specific issue, and do not hesitate to open any new issue (like better code, readability, modularity and best practice, performance, Android UI or even functionality enhancements...).

Please know that I am not a keen Android developer not a keen datascientist.

If you contribute, please consider that I can port this to some Android application on Play store, It will be 100% free for Android users on Play store and anywhere, although I can add ads to generate some coffee expenses :)

If you want to maintain the project with me, you can alwayse ask.

Please keep it fair if you want to deploy anywhere, ask for permission.

Sweet coding !
