//
//  ViewController.m
//  StockPrice_CoreML
//
//  Created by Jeff Tang on 3/17/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"
#import "Stock.h"


@interface ViewController ()

@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    Stock *stock = [[Stock alloc] init];

    double input[] = {
        0.40294855,
        0.39574954,
        0.39789235,
        0.39879138,
        0.40368535,
        0.41156033,
        0.41556879,
        0.41904324,
        0.42543786,
        0.42040193,
        0.42384258,
        0.42249741,
        0.4153998 ,
        0.41925279,
        0.41295281,
        0.40598363,
        0.40289448,
        0.44182321,
        0.45822208,
        0.44975226};

    

    NSError *error = nil;
    NSArray *shape = @[@20, @1, @1];
    MLMultiArray *mlMultiArray =  [[MLMultiArray alloc] initWithShape:(NSArray*)shape
                                                              dataType:MLMultiArrayDataTypeDouble
                                                                 error:&error] ;
    
    
    
    for (int i = 0; i < 20; i++) {
        [mlMultiArray setObject:[NSNumber numberWithDouble:input[i]] atIndexedSubscript:(NSInteger)i];
    }
    
    StockOutput *output = [stock predictionFromFeatures:[[StockInput alloc] initWithInput1:mlMultiArray]  error:&error];
    NSLog(@"output = %@", output.output1 );

}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
