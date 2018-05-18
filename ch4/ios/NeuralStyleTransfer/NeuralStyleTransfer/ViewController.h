//
//  ViewController.h
//  NeuralStyleTransfer
//
//  Created by Jeff Tang on 1/16/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController

+ (UIImage *) convertBitmapRGBA8ToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height;
@end

