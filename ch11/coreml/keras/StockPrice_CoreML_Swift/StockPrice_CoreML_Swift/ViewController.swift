//
//  ViewController.swift
//  StockPrice_CoreML_Swift
//
//  Created by Jeff Tang on 3/17/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {
    private let stock = Stock()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let input = [
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
            0.44975226]
        
        
        guard let mlMultiArray = try? MLMultiArray(shape:[20,1,1], dataType:MLMultiArrayDataType.double) else {
            fatalError("Unexpected runtime error. MLMultiArray")
        }
        for (index, element) in input.enumerated() {
            mlMultiArray[index] = NSNumber(floatLiteral: element)
        }
        
        guard let output = try? stock.prediction(input: StockInput(bidirectional_1_input:mlMultiArray)) else {
            return
        }
        
        print(output.activation_1_Identity)

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

