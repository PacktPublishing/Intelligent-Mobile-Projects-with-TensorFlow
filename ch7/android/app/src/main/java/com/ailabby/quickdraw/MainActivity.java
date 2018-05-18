package com.ailabby.quickdraw;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;




public class MainActivity extends AppCompatActivity implements Runnable {

    private static final String MODEL_FILE = "file:///android_asset/quickdraw_frozen_long_blacklist_strip_transformed.pb";
    private static final String CLASSES_FILE = "file:///android_asset/classes.txt";

    private static final String INPUT_NODE1 = "Reshape";
    private static final String INPUT_NODE2 = "Squeeze";
    private static final String OUTPUT_NODE1 = "dense/BiasAdd";
    private static final String OUTPUT_NODE2 = "ArgMax";

    private static final int CLASSES_COUNT = 345;
    private String[] mClasses = new String[CLASSES_COUNT];

    private static final int BATCH_SIZE = 8;


    private QuickDrawView mDrawView;
    private Button mButton;
    private TextView mTextView;
    private String mResult = "";
    private boolean mCanDraw = false;

    private TensorFlowInferenceInterface mInferenceInterface;

    public boolean canDraw() {
        return mCanDraw;
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mDrawView = findViewById(R.id.drawview);
        mButton = findViewById(R.id.button);
        mTextView = findViewById(R.id.textview);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mCanDraw = true;
                mButton.setText("Restart");
                mTextView.setText("");
                mDrawView.clearAllPointsAndRedraw();
            }
        });

        String actualFilename = CLASSES_FILE.split("file:///android_asset/")[1];
        BufferedReader br = null;
        int linenum = 0;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                mClasses[linenum++] = line;
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading classes file!" , e);
        }

    }

    public void run() {
        classifyDrawing();
    }

    private int count = 1;
    private synchronized  void classifyDrawing() {

        Log.d(">>>>>>>", ""+(count++));
        try {
            double normalized_points[] = normalizeScreenCoordinates();
                    // cat below
                    /* {
                    -0.06666669f, -0.15238096,  0.  ,
                    -0.05490196, -0.06190476,  0.  ,
                    0.03921568,  0.24761905,  0.  ,
                    -0.12941176,  0.01428571,  0.  ,
                    -0.04705882,  0.02857143,  0.  ,
                    -0.03529413,  0.05714285,  0.  ,
                    -0.02745098,  0.09523812,  0.  ,
                    0.        ,  0.06666666,  0.  ,
                    0.01176471,  0.08571428,  0.  ,
                    0.03137255,  0.06190473,  0.  ,
                    0.10588236,  0.09047621,  0.  ,
                    0.18431374,  0.02380949,  0.  ,
                    0.08235294, -0.01428568,  0.  ,
                    0.06274509, -0.03333336,  0.  ,
                    0.12941176, -0.12857139,  0.  ,
                    0.04313725, -0.10000002,  0.  ,
                    0.01176471, -0.08095235,  0.  ,
                    -0.00784314, -0.06666669,  0.  ,
                    -0.02745098, -0.05238095,  0.  ,
                    -0.0862745 , -0.06190476,  0.  ,
                    -0.03921568, -0.36190477,  0.  ,
                    -0.05490196,  0.14761905,  0.  ,
                    -0.01960784,  0.16190477,  0.  ,
                    -0.09803921,  0.02380952,  1.  ,
                    -0.21960786,  0.31428573,  0.  ,
                    -0.18823531, -0.03809524,  0.  ,
                    -0.08235294,  0.        ,  1.  ,
                    0.27058825,  0.15238094,  0.  ,
                    -0.20784315,  0.01904762,  0.  ,
                    -0.09019608,  0.05238092,  1.  ,
                    0.34117648,  0.        ,  0.  ,
                    -0.13725491,  0.07619047,  0.  ,
                    -0.05882353,  0.06190479,  1.  ,
                    0.53725493, -0.33333331,  0.  ,
                    0.18039215, -0.00952381,  0.  ,
                    0.10196078,  0.01904762,  0.  ,
                    0.01960784,  0.01428568,  1.  ,
                    -0.2980392 ,  0.03809524,  0.  ,
                    0.31372547,  0.10000002,  1.  ,
                    -0.32941175, -0.01904762,  0.  ,
                    0.14509803,  0.16190475,  0.  ,
                    0.02745098,  0.05714285,  1.  ,
                    -0.33333331, -0.38571429,  0.  ,
                    -0.0784314 ,  0.02380955,  0.  ,
                    -0.00784314,  0.01428568,  0.  ,
                    0.01176471,  0.03333336,  0.  ,
                    0.07450983,  0.01904762,  0.  ,
                    0.03529412, -0.01904762,  0.  ,
                    0.        , -0.03809524,  0.  ,
                    -0.0784314 , -0.02857143,  1.  ,
                    -0.04705882, -0.16190478,  0.  ,
                    -0.00392157,  0.08095238,  1};*/

            long total_points = normalized_points.length / 3;
            float[] floatValues  = new float[normalized_points.length*BATCH_SIZE];
            for (int i=0; i<normalized_points.length; i++) {
                for (int j=0; j<BATCH_SIZE; j++)
                    // model expects float not double
                    floatValues[j*normalized_points.length + i] = (float)normalized_points[i];
            }

            // model expects int64 not int32 (int in Java)
            long[] seqlen = new long[BATCH_SIZE];
            for (int i=0; i<BATCH_SIZE; i++)
                seqlen[i] = total_points;

            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

            mInferenceInterface.feed(INPUT_NODE1, floatValues, BATCH_SIZE, total_points, 3);
            mInferenceInterface.feed(INPUT_NODE2, seqlen, BATCH_SIZE);

            float[] logits = new float[CLASSES_COUNT * BATCH_SIZE];
            float[] argmax = new float[CLASSES_COUNT * BATCH_SIZE];

            mInferenceInterface.run(new String[] {OUTPUT_NODE1, OUTPUT_NODE2}, false);
            mInferenceInterface.fetch(OUTPUT_NODE1, logits);
            mInferenceInterface.fetch(OUTPUT_NODE1, argmax);

            // get sum of exp to normaliz the logits probabilities
            double sum = 0.0;
            for (int i=0; i<CLASSES_COUNT; i++)
                sum += Math.exp(logits[i]);

            List<Pair<Integer, Float>> prob_idx = new ArrayList<Pair<Integer, Float>>();
            for (int j = 0; j < CLASSES_COUNT; j++) {
                prob_idx.add(new Pair(j, (float)(Math.exp(logits[j]) / sum) ));
            }

            Collections.sort(prob_idx, new Comparator<Pair<Integer, Float>>() {
                @Override
                public int compare(final Pair<Integer, Float> o1, final Pair<Integer, Float> o2) {
                    return o1.second > o2.second ? -1 : (o1.second == o2.second ? 0 : 1);
                }
            });

            // get top 5 over threshold
            mResult = "";
            for (int i=0; i<5; i++) {
                //Log.d("", ""+prob_idx.get(i).second+","+prob_idx.get(i).first);
                if (prob_idx.get(i).second > 0.1) {
                    if (mResult == "") mResult = "" + mClasses[prob_idx.get(i).first];// + "(" + prob_idx.get(i).second + ")";
                    else mResult = mResult + ", " + mClasses[prob_idx.get(i).first];// + "(" + prob_idx.get(i).second + ")";
                }
            }

            Log.d(">>>>>", mResult);

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {
                            mTextView.setText(mResult);
                        }
                    });

        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }


    private double[] normalizeScreenCoordinates() {
        List<List<Pair<Float, Float>>> allPoints = mDrawView.getAllPoints();
        int total_points = 0;
        for (List<Pair<Float, Float>> cp : allPoints) {
            //Log.d(">>>>>>", "cp.size=" + cp.size());
            total_points += cp.size();
        }

        double[] normalized = new double[total_points * 3];

        float lowerx=Float.MAX_VALUE, lowery=Float.MAX_VALUE, upperx=-Float.MAX_VALUE, uppery=-Float.MAX_VALUE;
        for (List<Pair<Float, Float>> cp : allPoints) {
            for (Pair<Float, Float> p : cp) {
                //Log.d(">>>>>", ""+p.first + "," + p.second);

                if (p.first < lowerx) lowerx = p.first;
                if (p.second < lowery) lowery = p.second;
                if (p.first > upperx) upperx = p.first;
                if (p.second > uppery) uppery = p.second;
            }
        }
        float scalex = upperx - lowerx;
        float scaley = uppery - lowery;

        int n = 0;
        for (List<Pair<Float, Float>> cp : allPoints) {
            int m = 0;
            for (Pair<Float, Float> p : cp) {
                normalized[n*3] = (p.first - lowerx) / scalex;
                normalized[n*3+1] = (p.second - lowery) / scaley;
                normalized[n*3+2] = (m ==cp.size()-1 ? 1 : 0);
                n++; m++;
            }
        }

        for (int i=0; i<n-1; i++) {
            normalized[i*3] = normalized[(i+1)*3] - normalized[i*3];
            normalized[i*3+1] = normalized[(i+1)*3+1] - normalized[i*3+1];
            normalized[i*3+2] = normalized[(i+1)*3+2];
        }
        return normalized;
    }
}
