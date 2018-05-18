package com.ailabby.alphazero;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.HashMap;
import java.util.Random;
import java.util.Vector;

import static java.lang.Math.exp;
import static java.lang.Math.max;

public class MainActivity extends AppCompatActivity implements Runnable {

    private static final String MODEL_FILE = "file:///android_asset/alphazero19.pb";

    private static final String INPUT_NODE = "main_input";
    private static final String OUTPUT_NODE1 = "value_head/Tanh";
    private static final String OUTPUT_NODE2 = "policy_head/MatMul";

    private Button mButton;
    private BoardView mBoardView;
    private TextView mTextView;

    public static final int AI_PIECE = -1;
    public static final int HUMAN_PIECE = 1;
    private static final int PIECES_NUM = 42;

    private Boolean aiFirst = false;
    private Boolean aiTurn = false;

    private Vector<Integer> aiMoves = new Vector<>();
    private Vector<Integer> humanMoves = new Vector<>();

    private int board[] = new int[PIECES_NUM];
    private static final HashMap<Integer, String> PIECE_SYMBOL;
    static
    {
        PIECE_SYMBOL = new HashMap<Integer, String>();
        PIECE_SYMBOL.put(AI_PIECE, "X");
        PIECE_SYMBOL.put(HUMAN_PIECE, "O");
        PIECE_SYMBOL.put(0, "-");
    }

    private TensorFlowInferenceInterface mInferenceInterface;

    private final int winners[][] = {
        {0,1,2,3},
        {1,2,3,4},
        {2,3,4,5},
        {3,4,5,6},

        {7,8,9,10},
        {8,9,10,11},
        {9,10,11,12},
        {10,11,12,13},

        {14,15,16,17},
        {15,16,17,18},
        {16,17,18,19},
        {17,18,19,20},

        {21,22,23,24},
        {22,23,24,25},
        {23,24,25,26},
        {24,25,26,27},

        {28,29,30,31},
        {29,30,31,32},
        {30,31,32,33},
        {31,32,33,34},

        {35,36,37,38},
        {36,37,38,39},
        {37,38,39,40},
        {38,39,40,41},

        // vertically
        {0,7,14,21},
        {7,14,21,28},
        {14,21,28,35},

        {1,8,15,22},
        {8,15,22,29},
        {15,22,29,36},

        {2,9,16,23},
        {9,16,23,30},
        {16,23,30,37},

        {3,10,17,24},
        {10,17,24,31},
        {17,24,31,38},

        {4,11,18,25},
        {11,18,25,32},
        {18,25,32,39},

        {5,12,19,26},
        {12,19,26,33},
        {19,26,33,40},

        {6,13,20,27},
        {13,20,27,34},
        {20,27,34,41},

        // diagonally
        {3,9,15,21},

        {4,10,16,22},
        {10,16,22,28},

        {5,11,17,23},
        {11,17,23,29},
        {17,23,29,35},

        {6,12,18,24},
        {12,18,24,30},
        {18,24,30,36},
        {13,19,25,31},
        {19,25,31,37},
        {20,26,32,38},

        {3,11,19,27},

        {2,10,18,26},
        {10,18,26,34},

        {1,9,17,25},
        {9,17,25,33},
        {17,25,33,41},

        {0,8,16,24},
        {8,16,24,32},
        {16,24,32,40},
        {7,15,23,31},
        {15,23,31,39},
        {14,22,30,38}};


    public boolean getAITurn() {
        return aiTurn;
    }

    public boolean getAIFirst() {
        return aiFirst;
    }

    public Vector<Integer> getAIMoves() {
        return aiMoves;
    }

    public Vector<Integer> getHumanMoves() {
        return humanMoves;
    }

    public int[] getBoard() {
        return board;
    }

    public void setAiTurn() {
        aiTurn = true;
    }



    public boolean aiWon(int bd[]) {
        for (int i=0; i<69; i++) {
            int sum = 0;
            for (int j=0; j<4; j++)
                sum += bd[winners[i][j]];
            if (sum == 4*AI_PIECE ) return true;
        }
        return false;
    }

    public boolean  aiLost(int bd[]) {
        for (int i=0; i<69; i++) {
            int sum = 0;
            for (int j=0; j<4; j++)
                sum += bd[winners[i][j]];
            if (sum == 4*HUMAN_PIECE ) return true;
        }
        return false;
    }

    public boolean aiDraw(int bd[]) {
        boolean hasZero = false;
        for (int i=0; i<PIECES_NUM; i++) {
            if (bd[i] == 0) {
                hasZero = true;
                break;
            }
        }
        if (!hasZero) return true;
        return false;
    }


    public boolean gameEnded(int[] bd) {
        if (aiWon(bd) || aiLost(bd) || aiDraw(bd)) return true;

        return false;
    }

    void getAllowedActions(int bd[], Vector<Integer> actions) {

        for (int i=0; i<PIECES_NUM; i++) {
            if (i>=PIECES_NUM-7) {
                if (bd[i] == 0)
                    actions.add(i);
            }
            else {
                if (bd[i] == 0 && bd[i+7] != 0)
                    actions.add(i);
            }
        }

        for (int i=0; i<actions.size(); i++) {
            Log.d("<<<", "allowedAction: " +actions.get(i));
        }
    }

    public TextView getTextView() {
        return mTextView;
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mButton = findViewById(R.id.button);
        mTextView = findViewById(R.id.textview);
        mBoardView = findViewById(R.id.boardview);

        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mButton.setText("Replay");
                mTextView.setText("");

                Random rand = new Random();
                int n = rand.nextInt(2);

                aiFirst = (n==0); // make this random between true and false

                if (aiFirst) aiTurn = true;
                else aiTurn = false;

                if (aiTurn)
                    mTextView.setText("Waiting for AI's move");
                else
                    mTextView.setText("Tap the column for your move");

                for (int i=0; i<PIECES_NUM; i++)
                    board[i] = 0;
                aiMoves.clear();
                humanMoves.clear();
                mBoardView.drawBoard();

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });


    }

    @Override
    public void run() {

        final String result = playGame();

        runOnUiThread(
                new Runnable() {
                    @Override
                    public void run() {
                        mBoardView.invalidate();
                        mTextView.setText(result);
                    }
                });
    }

    void softmax(float vals[], int count) {
        float maxval = -Float.MAX_VALUE;
        for (int i=0; i<count; i++) {
            maxval = max(maxval, vals[i]);
        }
        float sum = 0.0f;
        for (int i=0; i<count; i++) {
            vals[i] = (float)exp(vals[i] - maxval);
            sum += vals[i];
        }
        for (int i=0; i<count; i++) {
            vals[i] /= sum;
        }
    }


    void getProbs(int binary[], float probs[]) {
        if (mInferenceInterface == null) {
            AssetManager assetManager = getAssets();
            mInferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
        }

        float[] floatValues  = new float[2*6*7];

        for (int i=0; i<2*6*7; i++) {
            floatValues[i] = binary[i];
        }

        float[] value = new float[1];
        float[] policy = new float[42];

        mInferenceInterface.feed(INPUT_NODE, floatValues, 1, 2, 6, 7);
        mInferenceInterface.run(new String[] {OUTPUT_NODE1, OUTPUT_NODE2}, false);
        mInferenceInterface.fetch(OUTPUT_NODE1, value);
        mInferenceInterface.fetch(OUTPUT_NODE2, policy);

//        Log.d(">>>", ""+value[0]);
//
//        for (int i = 0; i < 42; i++) {
//            Log.d(">>>", ""+policy[i]);
//        }

        Vector<Integer> actions = new Vector<>();
        getAllowedActions(board, actions);
        for (int action : actions) {
            probs[action] = policy[action];
        }

        softmax(probs, PIECES_NUM);
//        for (int i = 0; i < PIECES_NUM; i++) {
//            Log.d(">>>", "" + probs[i]);
//        }
    }

    void printBoard(int bd[]) {
        for (int i = 0; i<6; i++) {
            for (int j=0; j<7; j++) {
                System.out.print(" " + PIECE_SYMBOL.get(bd[i*7+j]));
            }
            System.out.println("");
        }

        System.out.println("\n\n");
    }


    String playGame() {
        if (!aiTurn) return "Tap the column for your move";

        int binary[] = new int[PIECES_NUM*2];

        // convert board to binary input
        for (int i=0; i<PIECES_NUM; i++)
            if (board[i] == 1) binary[i] = 1;
            else binary[i] = 0;

        for (int i=0; i<PIECES_NUM; i++)
            if (board[i] == -1) binary[42+i] = 1;
            else binary[PIECES_NUM+i] = 0;

        float probs[] = new float[PIECES_NUM];
        for (int i=0; i<PIECES_NUM; i++)
            probs[i] = -100.0f; // make all i's not one of the allowed actions 0 after softmax
        getProbs(binary, probs);
        int action = -1;
        // make the AI move with the highest prob
        float max = 0.0f;
        for (int i=0; i<PIECES_NUM; i++) {
            if (probs[i] > max) {
                max = probs[i];
                action = i;
            }
        }

        board[action] = AI_PIECE;
        printBoard(board);
        aiMoves.add(action);

        if (aiWon(board)) return "AI Won!";
        else if (aiLost(board)) return "You Won!";
        else if (aiDraw(board)) return "Draw";

        aiTurn = false;
        return "Tap the column for your move";

    }

}
