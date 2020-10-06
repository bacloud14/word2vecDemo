package com.bacloud.word2vecdemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Collectors;

public class MainActivity extends AppCompatActivity implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkDependencies();
        verifyStoragePermission(MainActivity.this);
        startBackgroundThread();
//        Model myNetwork = null;
//        String path ="";
//        saveNetLocal(myNetwork, path);
//        loadNetLocal(path);
//        INDArray input = null;
//        loadNetExternal(input);
    }

    private void startBackgroundThread() {
        Thread thread;
        thread = new Thread(() -> {
            createAndUseNetwork();
        });
        thread.start();
    }

    private final static int ngrams = 2;

    private void createAndUseNetwork() {
//        int seed = 248;
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(WeightInit.XAVIER)
//                .updater(Updater.ADAM)
//                .list()
//                .layer(new DenseLayer.Builder()
//                        .nIn(2)
//                        .nOut(3)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(new DenseLayer.Builder()
//                        .nOut(2)
//                        .activation(Activation.RELU)
//                        .build())
//                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) //NEGATIVELOGLIKELIHOOD
//                        .weightInit(WeightInit.XAVIER)
//                        .activation(Activation.SIGMOID)
//                        .nOut(1).build())
//                .build();
//
//
//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
        String filePath = "raw_sentences.txt";

        ArrayList<Collection<String>> emojisVectors = new ArrayList();

        String modelPath = "pathToWriteto.txt";
        File test = new File(this.getFilesDir().getAbsolutePath(), modelPath);
        Word2Vec vec = WordVectorSerializer.readWord2VecModel(test);

        try {
            FileInputStream fis = new FileInputStream("emojisVectors");
            ObjectInputStream ois = new ObjectInputStream(fis);
            emojisVectors = (ArrayList<Collection<String>>) ois.readObject();
            ois.close();
            fis.close();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            return;
        } catch (ClassNotFoundException c) {
            System.out.println("Class not found");
            c.printStackTrace();
            return;
        }
        String emojisPath = "emojis.json";
        JSONParser parser = new JSONParser();
        ArrayList<String[]> rows = new ArrayList<String[]>();
        try {
            JSONArray jsonArray = (JSONArray) parser.parse(new FileReader(emojisPath));

            for (int i = 0; i < jsonArray.size(); i++) {
                JSONObject jsonobject = (JSONObject) jsonArray.get(i);
                String[] row = new String[3];
                row[0] = (String) jsonobject.get("name");
                row[1] = (String) jsonobject.get("unicode");
                JSONArray keywords = (JSONArray) jsonobject.get("keywords");
                String row2 = "";
                for (int j = 0; j < keywords.size(); j++) {
                    row2 = row2 + " " + keywords.get(j);
                }
                row[2] = row2;
                rows.add(row);
            }
        } catch (ParseException | FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        DLUtils.check(vec, emojisVectors, rows, "star");
        DLUtils.check(vec, emojisVectors, rows, "nice people");
        DLUtils.check(vec, emojisVectors, rows, "black man");
        DLUtils.check(vec, emojisVectors, rows, "very excited");
        DLUtils.check(vec, emojisVectors, rows, "skateboard snow");
    }

    private String pathToString(String s) {
        String content = null;
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(s)));

            // do reading, usually loop until end of file reading
//            String mLine;
//            while ((mLine = reader.readLine()) != null) {
//                //process line
//            }
            content = reader.lines().collect(Collectors.joining());
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        return content;
    }


    private void loadNetExternal(INDArray input) {
        try {
            // Load name of model file (yourModel.zip).
            InputStream is = getResources().openRawResource(R.raw.your_model);
            // Load yourModel.zip.
            MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(is);
            // Use yourModel.

            INDArray results = restored.output(input);

            System.out.println("Results: " + results);
            // Handle the exception error
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MultiLayerNetwork loadNetLocal(String path) {
        MultiLayerNetwork restored = null;
        try {
            //Load the model
            File file = new File(Environment.getExternalStorageDirectory() + "/trained_model.zip");
            restored = ModelSerializer.restoreMultiLayerNetwork(file);

        } catch (Exception e) {
            Log.e("Load from External Storage error", e.getMessage());
        }
        return restored;
    }

    private void saveNetLocal(Model myNetwork, String path) {
        try {
            File file = new File(Environment.getExternalStorageDirectory() + "/trained_model.zip");
            OutputStream outputStream = new FileOutputStream(file);
            boolean saveUpdater = true;
            ModelSerializer.writeModel(myNetwork, outputStream, saveUpdater);

        } catch (Exception e) {
            Log.e("saveToExternalStorage error", e.getMessage());
        }
    }

    public void verifyStoragePermission(Activity activity) {
        // Get permission status
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission we request it
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE
            );
        }
    }

    private void checkDependencies() {
        // this will limit frequency of gc calls to 5000 milliseconds
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
    }


}