package com.ailabby.alphazero;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

/**
 * Created by jeffmbair on 3/14/18.
 */

public class BoardView extends View {
    private Path mPathBoard, mPathAIPieces, mPathHumanPieces;
    private Paint mPaint, mCanvasPaint;
    private Canvas mCanvas;
    private Bitmap mBitmap;
    private MainActivity mActivity;

    private static final float MARGINX = 20.0f;
    private static final float MARGINY = 210.0f;
    private float endY;
    private float columnWidth;

    public BoardView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mActivity = (MainActivity) context;

        setPathPaint();
    }

    private void setPathPaint() {
        mPathBoard = new Path();
        mPathAIPieces = new Path();
        mPathHumanPieces = new Path();
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setStrokeWidth(18);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mCanvasPaint = new Paint(Paint.DITHER_FLAG);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        mCanvas = new Canvas(mBitmap);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawBitmap(mBitmap, 0, 0, mCanvasPaint);
        columnWidth = (canvas.getWidth() - 2*MARGINX) / 7.0f;

        for (int i=0; i<8; i++) {
            float x = MARGINX + i * columnWidth;
            mPathBoard.moveTo(x, MARGINY);
            mPathBoard.lineTo(x, canvas.getHeight()-MARGINY);
        }
        Log.d(">>>", "canvas:" + canvas.getWidth() +","+canvas.getHeight());

        mPathBoard.moveTo(MARGINX, canvas.getHeight()-MARGINY);
        mPathBoard.lineTo(MARGINX + 7*columnWidth, canvas.getHeight()-MARGINY);
        mPaint.setColor(0xFF0000FF);
        canvas.drawPath(mPathBoard, mPaint);

        endY = canvas.getHeight()-MARGINY;
        int columnPieces[] = {0,0,0,0,0,0,0};

        if (mActivity.getAIFirst()) {
            for (int i=0; i<mActivity.getAIMoves().size(); i++) {
                int action = mActivity.getAIMoves().get(i);
                int column = action % 7;
                float x = MARGINX + column * columnWidth + columnWidth / 2.0f;
                float y = canvas.getHeight()-MARGINY-columnWidth*columnPieces[column]-columnWidth/2.0f;
                mPathAIPieces.addCircle(x,y, columnWidth/2, Path.Direction.CW);
                mPaint.setColor(0xFFFF0000);
                canvas.drawPath(mPathAIPieces, mPaint);
                columnPieces[column]++;

                if (i<mActivity.getHumanMoves().size()) {
                    action = mActivity.getHumanMoves().get(i);
                    column = action % 7;
                    x = MARGINX + column * columnWidth + columnWidth / 2.0f;
                    y = canvas.getHeight()-MARGINY-columnWidth*columnPieces[column]-columnWidth/2.0f;
                    mPathHumanPieces.addCircle(x,y, columnWidth/2, Path.Direction.CW);
                    mPaint.setColor(0xFFFFFF00);
                    canvas.drawPath(mPathHumanPieces, mPaint);

                    columnPieces[column]++;
                }
            }
        }
        else {
            for (int i=0; i<mActivity.getHumanMoves().size(); i++) {
                int action = mActivity.getHumanMoves().get(i);
                int column = action % 7;
                float x = MARGINX + column * columnWidth + columnWidth / 2.0f;
                float y = canvas.getHeight()-MARGINY-columnWidth*columnPieces[column]-columnWidth/2.0f;
                mPathHumanPieces.addCircle(x,y, columnWidth/2, Path.Direction.CW);
                mPaint.setColor(0xFFFFFF00);
                canvas.drawPath(mPathHumanPieces, mPaint);
                columnPieces[column]++;

                if (i<mActivity.getAIMoves().size()) {
                    action = mActivity.getAIMoves().get(i);
                    column = action % 7;
                    x = MARGINX + column * columnWidth + columnWidth / 2.0f;
                    y = canvas.getHeight()-MARGINY-columnWidth*columnPieces[column]-columnWidth/2.0f;
                    mPathAIPieces.addCircle(x,y, columnWidth/2, Path.Direction.CW);
                    mPaint.setColor(0xFFFF0000);
                    canvas.drawPath(mPathAIPieces, mPaint);

                    columnPieces[column]++;
                }
            }
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mActivity.getAITurn()) return true;

        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                break;
            case MotionEvent.ACTION_MOVE:
                break;
            case MotionEvent.ACTION_UP:
                if (y < MARGINY || y > endY) return true;

                int column = (int)((x-MARGINX)/columnWidth);

                for (int i=0; i<6; i++)
                    if (mActivity.getBoard()[35+column-7*i] == 0) {
                        mActivity.getBoard()[35+column-7*i] = MainActivity.HUMAN_PIECE;
                        mActivity.getHumanMoves().add(35+column-7*i);
                        break;
                    }

                invalidate();

                mActivity.setAiTurn();
                if (mActivity.gameEnded(mActivity.getBoard())) {
                    if (mActivity.aiWon(mActivity.getBoard()))
                        mActivity.getTextView().setText("AI Won!");
                    else if (mActivity.aiLost(mActivity.getBoard()))
                        mActivity.getTextView().setText("You Won!");
                    else if (mActivity.aiDraw(mActivity.getBoard()))
                        mActivity.getTextView().setText("Draw");
                    return true;
                }
                Thread thread = new Thread(mActivity);
                thread.start();

                break;
            default:
                return false;
        }

        return true;
    }


    public void drawBoard() {
        mBitmap = Bitmap.createBitmap(mBitmap.getWidth(), mBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        mCanvas = new Canvas(mBitmap);
        mCanvasPaint = new Paint(Paint.DITHER_FLAG);
        mCanvas.drawBitmap(mBitmap, 0, 0, mCanvasPaint);

        setPathPaint();

        invalidate();

    }

}
