package com.ailabby.fastneuraltransfer;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements Runnable {
    // use self-trained Fast Style Transfer model
//    private static final String MODEL_FILE = "file:///android_asset/fnn_starry_night_420x560.pb";
//    private static final String INPUT_NODE = "img_placeholder";
//    private static final String OUTPUT_NODE = "preds";

    // use pre-trained TensorFlow Magenta model
    private static final int NUM_STYLES = 26;
    private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";




    private static final String IMAGE_NAME = "www.jpg";
    private static final int WANTED_WIDTH = 420; //300;
    private static final int WANTED_HEIGHT = 560; //400;

    private ImageView mImageView;
    private Button mButton;
    private Bitmap mTransferredBitmap;

    private TensorFlowInferenceInterface mInferenceInterface;

    Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            mButton.setText("Style Transfer");
            String text = (String)msg.obj;
            Toast.makeText(MainActivity.this, text, Toast.LENGTH_SHORT).show();
            mImageView.setImageBitmap(mTransferredBitmap);
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
                mButton.setText("Processing...");
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
    }

    @Override
    public void run() {
        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];

        try {
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(IMAGE_NAME));
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, WANTED_WIDTH, WANTED_HEIGHT, true);
            scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3 + 0] = ((val >> 16) & 0x00FF);
                floatValues[i * 3 + 1] = ((val >> 8) & 0x00FF);
                floatValues[i * 3 + 2] = (val & 0x00FF);
            }

            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

            // use self-trained Fast Style Transfer model
//            mInferenceInterface.feed(INPUT_NODE, floatValues,  WANTED_HEIGHT, WANTED_WIDTH, 3);
//            mInferenceInterface.run(new String[] {OUTPUT_NODE}, false);
//            mInferenceInterface.fetch(OUTPUT_NODE, outputValues);
//            for (int i = 0; i < intValues.length; ++i) {
//                intValues[i] = 0xFF000000
//                        | (((int) outputValues[i * 3]) << 16)
//                        | (((int) outputValues[i * 3 + 1]) << 8)
//                        | ((int) outputValues[i * 3 + 2]);
//            }



            // use pre-trained TensorFlow Magenta model
            final float[] styleVals = new float[NUM_STYLES];
            for (int i = 0; i < NUM_STYLES; ++i) {
                styleVals[i] = 1.0f / NUM_STYLES;
            }
            //styleVals[19] = 1.0f;
            //styleVals[4] = 0.5f;
            mInferenceInterface.feed(INPUT_NODE, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 3);
            mInferenceInterface.feed("style_num", styleVals, NUM_STYLES);
            mInferenceInterface.run(new String[] {OUTPUT_NODE}, false);
            mInferenceInterface.fetch(OUTPUT_NODE, outputValues);
            for (int i = 0; i < intValues.length; ++i) {
                intValues[i] = 0xFF000000
                                | (((int) (outputValues[i * 3] * 255)) << 16)
                                | (((int) (outputValues[i * 3 + 1] * 255)) << 8)
                                | ((int) (outputValues[i * 3 + 2] * 255));
            }


            Bitmap outputBitmap = scaledBitmap.copy( scaledBitmap.getConfig() , true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            mTransferredBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap.getWidth(), bitmap.getHeight(), true);

            Message msg = new Message();
            msg.obj = "Tranfer Processing Done";
            mHandler.sendMessage(msg);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
}
