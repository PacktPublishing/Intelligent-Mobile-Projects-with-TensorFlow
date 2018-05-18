//
//  ViewController.m
//  AlphaZero
//
//  Created by Jeff Tang on 3/8/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <queue>
#include <fstream>
#include <random>
#include "tensorflow_utils.h"
#include <ctime>
#include <cstdlib>

using namespace std;

UIImageView *_iv;
UIButton *_btn;
UILabel *_lbl;
BOOL _modelLoaded = NO;

static NSString* MODEL_FILE = @"alphazero19"; // model from version0019.h5
static NSString* MODEL_FILE_TYPE = @"pb";

const int AI_PIECE = -1;
const int HUMAN_PIECE = 1;
const int PIECES_NUM = 42;
const float BOARD_COLUMN_WIDTH = 50.0;
const float BOARD_LINE_WIDTH = 6.0;
const int MCTS_SIMS = 10;

bool aiFirst = false;
bool aiTurn = false;
float startX, startY, endY;
vector<int> aiMoves;
vector<int> humanMoves;
map<int, string> PIECE_SYMBOL = {{AI_PIECE, "X"}, {HUMAN_PIECE, "O"}, {0, "-"}};
int board[PIECES_NUM];

bool withMCTS = false;
enum GAME_MODE {
    AUTO_PLAY, MANUAL_PLAY, AUTO_PLAY_MCTS, MANUAL_PLAY_MCTS
} gameMode = MANUAL_PLAY; //AUTO_PLAY;

//int board[] = {0,  1,  1, -1,  1, -1,  0,
//               0,  1, -1, -1, -1,  1,  0,
//               0,  1, -1,  1, -1,  1,  0,
//               0, -1, -1, -1,  1, -1,  0,
//               1,  1,  1, -1, -1, -1, -1,
//               1,  1,  1, -1,  1,  1, -1};



ViewController *vc;

unique_ptr<tensorflow::Session> tf_session;

string playGameAuto(bool withMCTS);
string playGameManual(bool withMCTS);
void getAllowedActions(int bd[], vector<int> &actions);
bool gameEnded(int bd[]);
bool aiWon(int bd[]);
bool aiLost(int bd[]);
bool aiDraw(int bd[]);

@interface ViewController ()
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    vc = self;
    // Do any additional setup after loading the view, typically from a nib.
    
    _btn = [UIButton buttonWithType:UIButtonTypeSystem];
    [_btn setTranslatesAutoresizingMaskIntoConstraints:NO];
    _btn.titleLabel.font = [UIFont systemFontOfSize:32];
    [_btn setTitle:@"Play" forState:UIControlStateNormal];
    [self.view addSubview:_btn];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:_btn
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:_btn
                                                                attribute:NSLayoutAttributeCenterY
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeBottom
                                                               multiplier:1
                                                                 constant:-80];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    _lbl = [[UILabel alloc] init];
    [_lbl setTranslatesAutoresizingMaskIntoConstraints:NO];
    [self.view addSubview:_lbl];
    
    NSLayoutConstraint *horizontal2 = [NSLayoutConstraint constraintWithItem:_lbl
                                                                   attribute:NSLayoutAttributeCenterX
                                                                   relatedBy:NSLayoutRelationEqual
                                                                      toItem:self.view
                                                                   attribute:NSLayoutAttributeCenterX
                                                                  multiplier:1
                                                                    constant:0];
    NSLayoutConstraint *vertical2 = [NSLayoutConstraint constraintWithItem:_lbl
                                                                 attribute:NSLayoutAttributeTop
                                                                 relatedBy:NSLayoutRelationEqual
                                                                    toItem:self.view
                                                                 attribute:NSLayoutAttributeTop
                                                                multiplier:1
                                                                  constant:80];
    [self.view addConstraint:horizontal2];
    [self.view addConstraint:vertical2];
    
    
    [_btn addTarget:self action:@selector(btnTapped:) forControlEvents:UIControlEventTouchUpInside];
    
    
    _iv = [[UIImageView alloc] init];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [_iv setTranslatesAutoresizingMaskIntoConstraints:NO];
    [self.view addSubview:_iv];
    
    NSLayoutConstraint *horizontal3 = [NSLayoutConstraint constraintWithItem:_iv
                                                                   attribute:NSLayoutAttributeCenterX
                                                                   relatedBy:NSLayoutRelationEqual
                                                                      toItem:self.view
                                                                   attribute:NSLayoutAttributeCenterX
                                                                  multiplier:1
                                                                    constant:0];
    NSLayoutConstraint *vertical3 = [NSLayoutConstraint constraintWithItem:_iv
                                                                 attribute:NSLayoutAttributeTop
                                                                 relatedBy:NSLayoutRelationEqual
                                                                    toItem:self.view
                                                                 attribute:NSLayoutAttributeTop
                                                                multiplier:1
                                                                  constant:0];
    NSLayoutConstraint *width = [NSLayoutConstraint constraintWithItem:_iv
                                                             attribute:NSLayoutAttributeWidth
                                                             relatedBy:NSLayoutRelationEqual
                                                                toItem:self.view
                                                             attribute:NSLayoutAttributeWidth
                                                            multiplier:1
                                                              constant:0];
    NSLayoutConstraint *height = [NSLayoutConstraint constraintWithItem:_iv
                                                              attribute:NSLayoutAttributeHeight
                                                              relatedBy:NSLayoutRelationEqual
                                                                 toItem:self.view
                                                              attribute:NSLayoutAttributeHeight
                                                             multiplier:1
                                                               constant:0];
    [self.view addConstraint:horizontal3];
    [self.view addConstraint:vertical3];
    [self.view addConstraint:width];
    [self.view addConstraint:height];
    
    srand((unsigned int)time(NULL));

}

- (IBAction)btnTapped:(id)sender {
    int n = rand() % 2;
    NSLog(@"btnTapped, n=%d", n);
    aiFirst = (n==0); // make this random between true and false
    
    if (aiFirst) aiTurn = true;
    else aiTurn = false;

    [_btn setTitle:@"Replay" forState:UIControlStateNormal];
    if (gameMode == MANUAL_PLAY || gameMode == MANUAL_PLAY_MCTS) {
        if (aiTurn)
            [_lbl setText:@"Waiting for AI's move"];
        else
            [_lbl setText:@"Tap the column for your move"];
    }
    else
        [_lbl setText:@"Auto play going on..."];
    
    // reset the game state and redraw the board
    for (int i=0; i<PIECES_NUM; i++)
        board[i] = 0;
    aiMoves.clear();
    humanMoves.clear();
    
    _iv.image = [self createBoardImageInRect:_iv.frame];

//    _iv.layer.masksToBounds = YES;
//    _iv.layer.borderColor = [UIColor redColor].CGColor;
//    _iv.layer.borderWidth = 2.0;
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        std::string result = (gameMode == AUTO_PLAY ? playGameAuto(withMCTS) : playGameManual(withMCTS));
        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *rslt = [NSString stringWithCString:result.c_str() encoding:[NSString defaultCStringEncoding]];
            [_lbl setText:rslt];
            _iv.image = [self createBoardImageInRect:_iv.frame];
            
        });
    });
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (UIImage *)createBoardImageInRect:(CGRect)rect
{
    int margin_y = 170;

    UIGraphicsBeginImageContextWithOptions(CGSizeMake(rect.size.width, rect.size.height), NO, 0.0);
    UIBezierPath *path = [UIBezierPath bezierPath];

    startX = (rect.size.width - 7*BOARD_COLUMN_WIDTH)/2.0;
    startY = rect.origin.y+margin_y+30;
    endY = rect.origin.y - margin_y + rect.size.height;
    for (int i=0; i<8; i++) {
        CGPoint point = CGPointMake(startX + i * BOARD_COLUMN_WIDTH, startY);
        [path moveToPoint:point];
        point = CGPointMake(startX + i * BOARD_COLUMN_WIDTH, endY);
        [path addLineToPoint:point];
    }
    
    CGPoint point = CGPointMake(startX, endY);
    [path moveToPoint:point];
    point = CGPointMake(rect.size.width - startX, endY);
    [path addLineToPoint:point];

    
    path.lineWidth = BOARD_LINE_WIDTH;
    [[UIColor blueColor] setStroke];
    [path stroke];
    
    // draw aiMoves and humanMoves alternatively
    int columnPieces[] = {0,0,0,0,0,0,0};
    
    if (aiFirst) {
        for (int i=0; i<aiMoves.size(); i++) {
            int action = aiMoves[i];
            int column = action % 7;
            CGRect r = CGRectMake(startX + column * BOARD_COLUMN_WIDTH, endY - BOARD_COLUMN_WIDTH - BOARD_COLUMN_WIDTH * columnPieces[column], BOARD_COLUMN_WIDTH, BOARD_COLUMN_WIDTH);
            UIBezierPath *path = [UIBezierPath bezierPathWithOvalInRect:r];
            UIColor *color = [UIColor redColor];
            [color setFill];
            [path fill];
            columnPieces[column]++;
            
            if (i<humanMoves.size()) {
                int action = humanMoves[i];
                int column = action % 7;
                CGRect r = CGRectMake(startX + column * BOARD_COLUMN_WIDTH, endY - BOARD_COLUMN_WIDTH - BOARD_COLUMN_WIDTH * columnPieces[column], BOARD_COLUMN_WIDTH, BOARD_COLUMN_WIDTH);
                UIBezierPath *path = [UIBezierPath bezierPathWithOvalInRect:r];
                UIColor *color = [UIColor yellowColor];
                [color setFill];
                [path fill];
                columnPieces[column]++;
            }
        }
    }
    else {
        for (int i=0; i<humanMoves.size(); i++) {
            int action = humanMoves[i];
            int column = action % 7;
            CGRect r = CGRectMake(startX + column * BOARD_COLUMN_WIDTH, endY - BOARD_COLUMN_WIDTH - BOARD_COLUMN_WIDTH * columnPieces[column], BOARD_COLUMN_WIDTH, BOARD_COLUMN_WIDTH);
            UIBezierPath *path = [UIBezierPath bezierPathWithOvalInRect:r];
            UIColor *color = [UIColor yellowColor];
            [color setFill];
            [path fill];
            columnPieces[column]++;
            
            if (i<aiMoves.size()) {
                int action = aiMoves[i];
                int column = action % 7;
                CGRect r = CGRectMake(startX + column * BOARD_COLUMN_WIDTH, endY - BOARD_COLUMN_WIDTH - BOARD_COLUMN_WIDTH * columnPieces[column], BOARD_COLUMN_WIDTH, BOARD_COLUMN_WIDTH);
                UIBezierPath *path = [UIBezierPath bezierPathWithOvalInRect:r];
                UIColor *color = [UIColor redColor];
                [color setFill];
                [path fill];
                columnPieces[column]++;
            }
        }
    }
    
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}


- (void) touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    if (aiTurn) return;
    
    UITouch *touch = [touches anyObject];
    CGPoint point = [touch locationInView:self.view];
    
    NSLog(@"touchesEnded: point:%f, %f", point.x, point.y);
    
    if (point.y < startY || point.y > endY) return;
    
    int column = (point.x-startX)/BOARD_COLUMN_WIDTH;
    NSLog(@">>>column=%d", column);
    
    for (int i=0; i<6; i++)
        if (board[35+column-7*i] == 0) {
            board[35+column-7*i] = HUMAN_PIECE;
            humanMoves.push_back(35+column-7*i);
            NSLog(@"human move: %d. should be one of the allowed actions", 35+column-7*i);
            break;
        }
    
    
    _iv.image = [self createBoardImageInRect:_iv.frame];
    aiTurn = true;
    if (gameEnded(board)) {
        if (aiWon(board)) _lbl.text = @"AI Won!";
        else if (aiLost(board)) _lbl.text = @"You Won!";
        else if (aiDraw(board)) _lbl.text = @"Draw";
        return;
    }
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        std::string result = (gameMode == AUTO_PLAY ? playGameAuto(withMCTS) : playGameManual(withMCTS));
        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *rslt = [NSString stringWithCString:result.c_str() encoding:[NSString defaultCStringEncoding]];
            [_lbl setText:rslt];
            _iv.image = [self createBoardImageInRect:_iv.frame];
        });
    });
    
}



@end

int winners[69][4] = {
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

void softmax(float vals[], int count) {
    float max = -FLT_MAX;
    for (int i=0; i<count; i++) {
        max = fmax(max, vals[i]);
    }
    float sum = 0.0;
    for (int i=0; i<count; i++) {
        vals[i] = exp(vals[i] - max);
        sum += vals[i];
    }
    for (int i=0; i<count; i++) {
        vals[i] /= sum;
    }
}

bool aiWon(int bd[]) {
    for (int i=0; i<69; i++) {
        int sum = 0;
        for (int j=0; j<4; j++)
            sum += bd[winners[i][j]];
        if (sum == 4*AI_PIECE ) return true;
    }
    return false;
}

bool aiLost(int bd[]) {
    for (int i=0; i<69; i++) {
        int sum = 0;
        for (int j=0; j<4; j++)
            sum += bd[winners[i][j]];
        if (sum == 4*HUMAN_PIECE ) return true;
    }
    return false;
}

bool aiDraw(int bd[]) {
    bool hasZero = false;
    for (int i=0; i<PIECES_NUM; i++) {
        if (bd[i] == 0) {
            hasZero = true;
            break;
        }
    }
    if (!hasZero) return true;
    return false;
}

bool gameEnded(int bd[]) {
    if (aiWon(bd) || aiLost(bd) || aiDraw(bd)) return true;
    
    return false;
}

void getAllowedActions(int bd[], vector<int> &actions) {
    
    for (int i=0; i<PIECES_NUM; i++) {
        if (i>=PIECES_NUM-7) {
            if (bd[i] == 0)
                actions.push_back(i);
        }
        else {
            if (bd[i] == 0 && bd[i+7] != 0)
                actions.push_back(i);
        }
    }
    
    for (int i=0; i<actions.size(); i++) {
        cout << "allowedAction: " << actions[i] << endl;
    }
}


bool getProbs(int *binary, float *probs) {
    std::string input_name = "main_input";
    
    // The value v is a scalar evaluation, estimating the probability of the current player winning from position s.
    std::string output_name1 = "value_head/Tanh";
    
    // probabilities p represents the probability of selecting each move
    std::string output_name2 = "policy_head/MatMul";
    
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,2,6,7}));
    auto input_mapped = input_tensor.tensor<float, 4>();
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j<6; j++) {
            for (int k=0; k<7; k++) {
                input_mapped(0,i,j,k) = binary[i*42+j*7+k];
            }
        }
    }
    
    
    std::vector<tensorflow::Tensor> outputs;
    
    tensorflow::Status run_status = tf_session->Run({{input_name, input_tensor}},
                                                    {output_name1, output_name2}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Getting model failed:" << run_status;
        return false;
    }
    
    tensorflow::string status_string = run_status.ToString();
    
    tensorflow::Tensor* value_tensor = &outputs[0];
    tensorflow::Tensor* policy_tensor = &outputs[1];
    
    
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& value = value_tensor->flat<float>();
    long count = value.size();
    printf("value count=%ld, value=%f\n", count, value(0));
    
    
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& policy = policy_tensor->flat<float>();
    
    
    count = policy.size();
    printf("policy count=%ld\n", count); // should be the same as PIECES_NUM
    
    vector<int> actions;
    getAllowedActions(board, actions);
    for (int action : actions) {
        probs[action] = policy(action);
    }
    
    softmax(probs, PIECES_NUM);
    for (int i = 0; i < count; i++) {
        printf("%f\n", probs[i]);
    }
    
    return true;
}

void printBoard(int bd[]) {
    for (int i = 0; i<6; i++) {
        for (int j=0; j<7; j++) {
            cout << PIECE_SYMBOL[bd[i*7+j]] << " ";
        }
        cout << endl;
    }
    
    cout << endl << endl;
}

string playGameAuto(bool withMCTS) {
    if (!_modelLoaded) {
        tensorflow::Status load_status;
        
        load_status = LoadModel(MODEL_FILE, MODEL_FILE_TYPE, &tf_session);
        
        if (!load_status.ok()) {
            LOG(FATAL) << "Couldn't load model: " << load_status;
            return "";
        }
        _modelLoaded = YES;
    }
    
    
    int binary[PIECES_NUM*2];
    
    // initial board position
    for (int i=0; i<PIECES_NUM; i++)
        board[i] = 0;
    
    while (!gameEnded(board)) {

        // convert board to binary input
        for (int i=0; i<PIECES_NUM; i++)
            if (board[i] == 1) binary[i] = 1;
            else binary[i] = 0;
        
        for (int i=0; i<PIECES_NUM; i++)
            if (board[i] == -1) binary[42+i] = 1;
            else binary[PIECES_NUM+i] = 0;
        
        for (int i=0; i<PIECES_NUM*2; i++)
            printf("%d ", binary[i]);
    
        
        float *probs = new float[PIECES_NUM];
        for (int i=0; i<PIECES_NUM; i++)
            probs[i] = -100.0; // make all i's not one of the allowed actions 0 after softmax
        if (getProbs(binary, probs)) {
            // make the AI move with the highest prob
            float max = 0.0;
            int action = -1;
            for (int i=0; i<PIECES_NUM; i++) {
                if (probs[i] > max) {
                    max = probs[i];
                    action = i;
                    cout << "max=" << max << ",action=" << action << endl;
                }
            }
            board[action] = AI_PIECE;
            printBoard(board);
            aiMoves.push_back(action);
            
            delete []probs;
            
            dispatch_async(dispatch_get_main_queue(), ^{
                _iv.image = [vc createBoardImageInRect:_iv.frame];
                
            });
            sleep(1);
            
            if (gameEnded(board)) break;
            
            vector<int> actions;
            getAllowedActions(board, actions);
            if (actions.size() == 0) {
                cout << "No possible moves" << endl;
                return "Draw";
            }
            
            // make a random human legal (allowed) move
            int n = rand() % actions.size();
            cout << "***** allowed action size: " << actions.size() << ", n=" << n << endl;

            board[actions[n]] = HUMAN_PIECE;
            humanMoves.push_back(actions[n]);
            printBoard(board);
            
            dispatch_async(dispatch_get_main_queue(), ^{
                _iv.image = [vc createBoardImageInRect:_iv.frame];
                
            });
            sleep(1);
        }
        else { // running the model failed
            delete []probs;
        }
        
    }
    
    if (aiWon(board)) return "AI Won!";
    else if (aiLost(board)) return "You Won!";
    else return "Draw";
}

string board_string(int bd[]) {
    string s = "";
    for (int i=0; i<PIECES_NUM; i++)
        s += to_string(bd[i]);
    
    return s;
}

string playGameManual(bool withMCTS) {
    if (!_modelLoaded) {
        tensorflow::Status load_status;
        
        load_status = LoadModel(MODEL_FILE, MODEL_FILE_TYPE, &tf_session);
        
        if (!load_status.ok()) {
            LOG(FATAL) << "Couldn't load model: " << load_status;
            return "";
        }
        _modelLoaded = YES;
    }
    
    if (!aiTurn) return "Tap the column for your move";
    
    int binary[PIECES_NUM*2];
    
    // convert board to binary input
    for (int i=0; i<PIECES_NUM; i++)
        if (board[i] == 1) binary[i] = 1;
        else binary[i] = 0;
    
    for (int i=0; i<PIECES_NUM; i++)
        if (board[i] == -1) binary[42+i] = 1;
        else binary[PIECES_NUM+i] = 0;
    
//    string s = "000000000000000000000000000000000001000100000000000000000000000000000000010000001000";
//    for (int i=0; i<PIECES_NUM*2; i++)
//        binary[i] = s[i]-48;
    
    for (int i=0; i<PIECES_NUM*2; i++)
        printf("%d", binary[i]);
    
    float *probs = new float[PIECES_NUM];
    for (int i=0; i<PIECES_NUM; i++)
        probs[i] = -100.0; // make all i's not one of the allowed actions 0 after softmax
    if (getProbs(binary, probs)) {
        int action = -1;
        if (withMCTS) {
            // run MCTS and return an action:
            map<string, map<string, float>> stats; // each board position's stat has visit_count, value, prob, parent (board position), and maybe UCT.
            
            // run random simulation
            for (int i=0; i<MCTS_SIMS; i++) {
                map<string, string> parent;
                int simulated_board[PIECES_NUM];
                for (int j=0; j<PIECES_NUM; j++) {
                    simulated_board[j] = board[j];
                }

                if (stats.find(board_string(simulated_board)) == stats.end()) {
                    map<string, float> stat;
                    stat["visit_count"] = 1.0;
                    stat["value"] = 0.0;
                    stats[board_string(simulated_board)] = stat;
                }
                else {
                    stats[board_string(simulated_board)]["visit_count"]++;
                }
                
                vector<string> parents; // each element is the parent of its following element
                parents.push_back(board_string(simulated_board));
                
                
                bool simulated_aiTurn = true;
                while (!gameEnded(simulated_board)) {
                    vector<int> actions;
                    getAllowedActions(simulated_board, actions);
                    if (actions.size() > 0) {
                        int n = rand() % actions.size();
                        simulated_board[actions[n]] = simulated_aiTurn ? AI_PIECE : HUMAN_PIECE;
                        simulated_aiTurn = !simulated_aiTurn;
                        printBoard(simulated_board);
                        
                        parents.push_back(board_string(simulated_board));
                        parent[board_string(simulated_board)] = parents[parents.size()-2];

                        map<string, float> stat;
                        if (stat.find("visit_count") != stat.end())
                            stat["visit_count"] += 1.0;
                        else
                            stat["visit_count"] = 1.0;
                        if (stat.find("value") == stat.end())
                            stat["value"] = 0.0;
                        stats[board_string(simulated_board)] = stat;
                    }
                }
                if (aiWon(simulated_board))
                    stats[board_string(simulated_board)]["value"] = 1.0;
                else if (aiLost(simulated_board))
                    stats[board_string(simulated_board)]["value"] = -1.0;
                else
                    stats[board_string(simulated_board)]["value"] = 0.5;
                
                // backprop the evaluated value through tht MCST
                string p = board_string(simulated_board);
                while (parent.find(p) != parent.end()) {
                    string q = parent[p];
                    stats[q]["value"] += stats[p]["value"];
                    p = q;
                }
            }
            
            for (map<string, map<string, float>>::iterator it = stats.begin(); it != stats.end(); it++) {
                cout << it->first << endl;
                map<string, float> stat = it->second;
                for (map<string, float>::iterator it2 = stat.begin(); it2 != stat.end(); it2++) {
                    cout << it2->first << ": " << it2->second << endl;
                }
            }
        

            
            // 2. get action values
            
            
            // 3. find the best action
            action = 0;
            
        }
        else {
            // make the AI move with the highest prob
            float max = 0.0;
            for (int i=0; i<PIECES_NUM; i++) {
                if (probs[i] > max) {
                    max = probs[i];
                    action = i;
                    cout << "max=" << max << ",action=" << action << endl;
                }
            }
        }
        
        // take the action
        board[action] = AI_PIECE;
        printBoard(board);
        aiMoves.push_back(action);

        delete []probs;
        
        if (aiWon(board)) return "AI Won!";
        else if (aiLost(board)) return "You Won!";
        else if (aiDraw(board)) return "Draw";
        
    } else {
        delete []probs;
    }

    
    aiTurn = false;
    return "Tap the column for your move";
}

