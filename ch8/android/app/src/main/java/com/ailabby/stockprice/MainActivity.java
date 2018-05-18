package com.ailabby.stockprice;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedInputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class MainActivity extends AppCompatActivity implements Runnable {

    private static final String TF_MODEL_FILENAME = "file:///android_asset/amzn_tf_frozen.pb";
    private static final String KERAS_MODEL_FILENAME = "file:///android_asset/amzn_keras_frozen.pb";
    private static final String INPUT_NODE_NAME_TF = "Placeholder";
    private static final String OUTPUT_NODE_NAME_TF = "preds";
    private static final String INPUT_NODE_NAME_KERAS = "bidirectional_1_input";
    private static final String OUTPUT_NODE_NAME_KERAS = "activation_1/Identity";
    private static final int SEQ_LEN = 20;
    private static final float LOWER = 5.97f;
    private static final float SCALE = 1479.37f;

    private TensorFlowInferenceInterface mInferenceInterface;

    private Button mButtonTF;
    private Button mButtonKeras;
    private TextView mTextView;
    private boolean mUseTFModel;
    private String mResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButtonTF = findViewById(R.id.tfbutton);
        mButtonKeras = findViewById(R.id.kerasbutton);
        mTextView = findViewById(R.id.textview);
        mTextView.setMovementMethod(new ScrollingMovementMethod());
        mButtonTF.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mUseTFModel = true;
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
        mButtonKeras.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mUseTFModel = false;
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

    }


    @Override
    public void run() {
        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mTextView.setText("Getting data...");
                    }
                });


        float[] floatValues  = new float[SEQ_LEN];

        try {
            URL url = new URL("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=amzn&apikey=4SOSJM2XCRIB5IUS&datatype=csv&outputsize=compact");
            HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
            InputStream in = new BufferedInputStream(urlConnection.getInputStream());
            Scanner s = new Scanner(in).useDelimiter("\\n");
            mResult = "Last 20 Days:\n";
            if (s.hasNext()) s.next(); // get rid of the first title line
            List<String> priceList = new ArrayList<>();
            while (s.hasNext()) {
                String line = s.next();
                String[] items = line.split(",");
                priceList.add(items[4]);
            }

            for (int i=0; i<SEQ_LEN; i++)
                mResult += priceList.get(SEQ_LEN-i-1) + "\n";
            Log.d(">>>>", mResult);


            for (int i=0; i<SEQ_LEN; i++) {
                if (mUseTFModel)
                    floatValues[i] =  Float.parseFloat(priceList.get(SEQ_LEN-i-1));
                else
                    floatValues[i] = (Float.parseFloat(priceList.get(SEQ_LEN-i-1)) - LOWER) / SCALE;
            }

//        float[] floatValues  = new float[] {
//                0.40294855f,
//                0.39574954f,
//                0.39789235f,
//                0.39879138f,
//                0.40368535f,
//                0.41156033f,
//                0.41556879f,
//                0.41904324f,
//                0.42543786f,
//                0.42040193f,
//                0.42384258f,
//                0.42249741f,
//                0.4153998f,
//                0.41925279f,
//                0.41295281f,
//                0.40598363f,
//                0.40289448f,
//                0.44182321f,
//                0.45822208f,
//                0.44975226f};

//        float[] floatValues  = new float[] {761.01f,
//                761.09f,
//                769.69f,
//                778.52f,
//                775.1f,
//                780.22f,
//                789.74f,
//                804.7f,
//                805.75f,
//                799.16f,
//                816.11f,
//                828.72f,
//                829.05f,
//                837.31f,
//                836.74f,
//                834.03f,
//                844.36f,
//                841.66f,
//                839.43f,
//                841.71f};
//    };




            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, mUseTFModel ? TF_MODEL_FILENAME : KERAS_MODEL_FILENAME);

            mInferenceInterface.feed(mUseTFModel ? INPUT_NODE_NAME_TF : INPUT_NODE_NAME_KERAS, floatValues, 1, SEQ_LEN, 1);

            float[] predictions = new float[mUseTFModel ? SEQ_LEN : 1];

            mInferenceInterface.run(new String[] {mUseTFModel ? OUTPUT_NODE_NAME_TF : OUTPUT_NODE_NAME_KERAS}, false);
            mInferenceInterface.fetch(mUseTFModel ? OUTPUT_NODE_NAME_TF : OUTPUT_NODE_NAME_KERAS, predictions);
            if (mUseTFModel) {
                mResult += "\nPrediction with TF RNN model:\n" + predictions[SEQ_LEN - 1];
                Log.d(">>>", "" + predictions[SEQ_LEN - 1]);
            }
            else {
                mResult += "\nPrediction with Keras RNN model:\n" + (predictions[0] * SCALE + LOWER);
                Log.d(">>>", "" + (predictions[0]) * SCALE + LOWER);
            }

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            mTextView.setText(mResult);

                        }
                    });

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }


    }


}
