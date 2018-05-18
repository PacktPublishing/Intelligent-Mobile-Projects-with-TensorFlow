package com.ailabby.gan;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements Runnable {

    private static final String MODEL_FILE1 = "file:///android_asset/gan_mnist.pb";
    private static final String MODEL_FILE2 = "file:///android_asset/pix2pix_transformed_quantized.pb";

    private static final String INPUT_NODE1 = "z_placeholder";
    private static final String OUTPUT_NODE1 = "Sigmoid_1";
    private static final String INPUT_NODE2 = "image_feed";
    private static final String OUTPUT_NODE2 = "generator_1/deprocess/truediv";

    private static final String IMAGE_NAME = "ww.png";
    private static final int WANTED_WIDTH = 256;
    private static final int WANTED_HEIGHT = 256;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;

    private static final int BATCH_SIZE = 1;

    private Button mButtonMNIST;
    private Button mButtonPix2Pix;
    private ImageView mImageView;
    private Bitmap mGeneratedBitmap;
    private boolean mMNISTModel;

    private TensorFlowInferenceInterface mInferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButtonMNIST = findViewById(R.id.mnistbutton);
        mButtonPix2Pix = findViewById(R.id.pix2pixbutton);
        mImageView = findViewById(R.id.imageview);
        try {
            AssetManager am = getAssets();
            InputStream is = am.open(IMAGE_NAME);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            mImageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }

        mButtonMNIST.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mMNISTModel = true;
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
        mButtonPix2Pix.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    AssetManager am = getAssets();
                    InputStream is = am.open(IMAGE_NAME);
                    Bitmap bitmap = BitmapFactory.decodeStream(is);
                    mImageView.setImageBitmap(bitmap);
                    mMNISTModel = false;
                    Thread thread = new Thread(MainActivity.this);
                    thread.start();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }

    @Override
    public void run() {
        if (mMNISTModel)
            runMNISTModel();
        else
            runPix2PixBlurryModel();
    }

    void runMNISTModel() {
        float[] floatValues  = new float[BATCH_SIZE*100];

        Random r = new Random();
        for (int i=0; i<BATCH_SIZE; i++) {
            for (int j=0; i<100; i++) {
                double sample = r.nextGaussian();
                floatValues[i] = (float)sample;
                Log.d(">>>", ""+sample);
            }
        }


        float[] outputValues = new float[BATCH_SIZE * 28 * 28];
        AssetManager assetManager = getAssets();
        mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE1);


        mInferenceInterface.feed(INPUT_NODE1, floatValues, BATCH_SIZE, 100);
        mInferenceInterface.run(new String[] {OUTPUT_NODE1}, false);
        mInferenceInterface.fetch(OUTPUT_NODE1, outputValues);

        int[] intValues = new int[BATCH_SIZE * 28 * 28];
        for (int i = 0; i < intValues.length; i++) {
            intValues[i] = (int) (outputValues[i] * 255);
        }

        try {
            Bitmap bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
            for (int y=0; y<28; y++) {
                for (int x=0; x<28; x++) {
                    int c = intValues[y*28 + x];
                    //Log.d(">>>>", ""+c);

                    int color = (255 & 0xff) << 24 | (c & 0xff) << 16 | (c & 0xff) << 8 | (c & 0xff);
                    bitmap.setPixel(x, y, color);
                }

            }

            mGeneratedBitmap = Bitmap.createBitmap(bitmap);
        }
        catch (Exception e) {
            e.printStackTrace();
        }


            runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mImageView.setImageBitmap(mGeneratedBitmap);

                    }
                });
    }


    void runPix2PixBlurryModel() {
        int[] intValues = new int[WANTED_WIDTH * WANTED_HEIGHT];
        float[] floatValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];
        float[] outputValues = new float[WANTED_WIDTH * WANTED_HEIGHT * 3];

        try {
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(IMAGE_NAME));
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, WANTED_WIDTH, WANTED_HEIGHT, true);
            scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
            Log.d("****", ""+scaledBitmap.getWidth() +","+scaledBitmap.getHeight());
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];

                floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            }

            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE2);


            mInferenceInterface.feed(INPUT_NODE2, floatValues, 1, WANTED_HEIGHT, WANTED_WIDTH, 3);
            mInferenceInterface.run(new String[] {OUTPUT_NODE2}, false);
            mInferenceInterface.fetch(OUTPUT_NODE2, outputValues);

            for (int i = 0; i < intValues.length; ++i) {
                intValues[i] = 0xFF000000
                        | (((int) (outputValues[i * 3] * 255)) << 16)
                        | (((int) (outputValues[i * 3 + 1] * 255)) << 8)
                        | ((int) (outputValues[i * 3 + 2] * 255));
            }


            Bitmap outputBitmap = scaledBitmap.copy( scaledBitmap.getConfig() , true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            mGeneratedBitmap = Bitmap.createScaledBitmap(outputBitmap, bitmap.getWidth(), bitmap.getHeight(), true);

        }
        catch (Exception e) {
            e.printStackTrace();
        }

        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mImageView.setImageBitmap(mGeneratedBitmap);
                    }
                });


    }
}
