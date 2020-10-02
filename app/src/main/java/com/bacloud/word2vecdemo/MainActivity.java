package com.bacloud.word2vecdemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

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
        try {
            SentenceIterator iter = new BasicLineIterator(filePath);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
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