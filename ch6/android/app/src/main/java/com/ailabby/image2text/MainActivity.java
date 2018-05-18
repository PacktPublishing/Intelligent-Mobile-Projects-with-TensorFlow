package com.ailabby.image2text;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class MainActivity extends AppCompatActivity implements Runnable {

    private static final String MODEL_FILE = "file:///android_asset/image2text_frozen_transformed.pb";
    private static final String VOCAB_FILE = "file:///android_asset/word_counts.txt";
    private static final String IMAGE_NAME = "im2txt1.png";

    private static final String INPUT_NODE1 = "convert_image/Cast";
    private static final String OUTPUT_NODE1 = "lstm/initial_state";
    private static final String INPUT_NODE2 = "input_feed";
    private static final String INPUT_NODE3 = "lstm/state_feed";
    private static final String OUTPUT_NODE2 = "softmax";
    private static final String OUTPUT_NODE3 = "lstm/state";

    private static final int IMAGE_WIDTH = 299;
    private static final int IMAGE_HEIGHT = 299;
    private static final int IMAGE_CHANNEL = 3;

    private static final int CAPTION_LEN = 20;
    private static final int WORD_COUNT = 12000;
    private static final int STATE_COUNT = 1024;
    private static final int START_ID = 2;
    private static final int END_ID = 3;

    private String[] mWords = new String[WORD_COUNT];

    private ImageView mImageView;
    private Button mButton;

    //private HashMap<Integer, String> mIdToWord = new HashMap<>();

    private TensorFlowInferenceInterface mInferenceInterface;

    private int[] intValues;
    private float[] floatValues;

    Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            mButton.setText("Describe Me");
            String text = (String)msg.obj;
            Toast.makeText(MainActivity.this, text, Toast.LENGTH_LONG).show();
            mButton.setEnabled(true);
        } };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.imageview);
        try {
            AssetManager am = getAssets();
            InputStream is = am.open(IMAGE_NAME);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            mImageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }

        mButton = findViewById(R.id.button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mButton.setEnabled(false);
                mButton.setText("Processing...");
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });


        // read vocaburay file and do basic parsing
        String actualFilename = VOCAB_FILE.split("file:///android_asset/")[1];
        BufferedReader br = null;
        int linenum = 0;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                String word = line.split(" ")[0];
                mWords[linenum++] = word;
                //mIdToWord.put(linenum++, word);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading vocab file!" , e);
        }

    }

    @Override
    public void run() {
        try {
            intValues = new int[IMAGE_WIDTH * IMAGE_HEIGHT];
            floatValues = new float[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL];

            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(IMAGE_NAME));
            Bitmap croppedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT, true);
            croppedBitmap.getPixels(intValues, 0, IMAGE_WIDTH, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * IMAGE_CHANNEL + 0] = ((val >> 16) & 0xFF);
                floatValues[i * IMAGE_CHANNEL + 1] = ((val >> 8) & 0xFF);
                floatValues[i * IMAGE_CHANNEL + 2] = (val & 0xFF);
//                floatValues[i * IMAGE_CHANNEL + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
//                floatValues[i * IMAGE_CHANNEL + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
//                floatValues[i * IMAGE_CHANNEL + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            }


            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
            float[] initialState = new float[STATE_COUNT];

            mInferenceInterface.feed(INPUT_NODE1, floatValues, IMAGE_WIDTH,
                    IMAGE_HEIGHT, 3);
            mInferenceInterface.run(new String[] {OUTPUT_NODE1}, false);
            mInferenceInterface.fetch(OUTPUT_NODE1, initialState);


            long[] inputFeed = new long[] {START_ID};
            float[] stateFeed = new float[STATE_COUNT * inputFeed.length];
            for (int i=0; i < STATE_COUNT; i++) {
                //Log.d("!!!", ""+initialState[i]);
                stateFeed[i] = initialState[i];
            }

            List<Pair<Integer, Float>> captions = new ArrayList<Pair<Integer, Float>>();
            for (int i=0; i<CAPTION_LEN; i++) {
                // return probabilities for all words in the voc
                float[] softmax = new float[WORD_COUNT * inputFeed.length];
                float[] newstate = new float[STATE_COUNT * inputFeed.length];

                mInferenceInterface.feed(INPUT_NODE2, inputFeed, 1);
                mInferenceInterface.feed(INPUT_NODE3, stateFeed, 1, STATE_COUNT);

                mInferenceInterface.run(new String[]{OUTPUT_NODE2, OUTPUT_NODE3}, false);

                // will cause error  java.nio.BufferOverflowException if softmax is not large enough
                mInferenceInterface.fetch(OUTPUT_NODE2, softmax);
                mInferenceInterface.fetch(OUTPUT_NODE3, newstate);

                List<Pair<Integer, Float>> prob_id = new ArrayList<Pair<Integer, Float>>();
                for (int j = 0; j < WORD_COUNT; j++) {
                    prob_id.add(new Pair(j, softmax[j]));
                }

                Collections.sort(prob_id, new Comparator<Pair<Integer, Float>>() {
                    @Override
                    public int compare(final Pair<Integer, Float> o1, final Pair<Integer, Float> o2) {
                        return o1.second > o2.second ? -1 : (o1.second == o2.second ? 0 : 1);
                    }
                });

                if (prob_id.get(0).first == END_ID) break;

                captions.add(new Pair(prob_id.get(0).first, prob_id.get(0).first));

                inputFeed = new long[] {prob_id.get(0).first};
                for (int j=0; j < STATE_COUNT; j++) {
                    stateFeed[j] = newstate[j];
                }
            }

            String sentence = "";
            for (int i=0; i<captions.size(); i++) {
                if (captions.get(i).first == START_ID) continue;
                if (captions.get(i).first == END_ID) break;

                //sentence = sentence + " " + mIdToWord.get(captions.get(i).first);
                sentence = sentence + " " + mWords[captions.get(i).first];
                Log.d("captions", ""+captions.get(i).first);
            }
            Log.d("Description: ", sentence);


            Message msg = new Message();
            msg.obj = sentence;
            mHandler.sendMessage(msg);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
}
