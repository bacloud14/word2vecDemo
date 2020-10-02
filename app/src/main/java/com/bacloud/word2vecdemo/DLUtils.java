package com.bacloud.word2vecdemo;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.guava.base.Splitter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;


public class DLUtils {

    private final static int topNearest = 2;

    static void check(Word2Vec vec, ArrayList<Collection<String>> emojisVectors, ArrayList<String[]> rows, String sentence) {
        System.out.println("\n\nSentence: " + sentence);
        String sentence_ = String.join(" ", vec.wordsNearest(sentence.trim(), topNearest));
        if (sentence_.equals(""))
            sentence_ = sentence;
        double max = 0;
        int idx = -1;
        int det = -1;
        Collection<String> bestEmojiVector = emojisVectors.get(0);
        for (Collection<String> emojiVector : emojisVectors) {
            idx++;
            if (emojiVector.isEmpty())
                continue;
            double score = cosineSimForSentence(vec, String.join(" ", emojiVector), sentence_);
            if (score > max) {
                max = score;
                bestEmojiVector = emojiVector;
                det = idx;
            }

        }
        System.out.println("det " + det);
        System.out.println("bestEmojiVector " + bestEmojiVector.toString());
        System.out.print("row " + Arrays.toString(rows.get(det)));
        System.out.print("max " + max);
    }

    public static double cosineSimForSentence(Word2Vec vector, String sentence1, String sentence2) {
        Collection<String> label1 = Splitter.on(' ').splitToList(sentence1);
        Collection<String> label2 = Splitter.on(' ').splitToList(sentence2);
        try {
            return Transforms.cosineSim(vector.getWordVectorsMean(label1), vector.getWordVectorsMean(label2));
        } catch (Exception e) {
            String exceptionMessage = e.getMessage();
            System.out.print(exceptionMessage);
        }
        return Transforms.cosineSim(vector.getWordVectorsMean(label1), vector.getWordVectorsMean(label2));

    }
}
