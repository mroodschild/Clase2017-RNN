/*
 * The MIT License
 *
 * Copyright 2017 Matías Roodschild <mroodschild@gmail.com>.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package org.gitia.clase2017.rnn.tp2;

import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.Feedforward;
import org.gitia.froog.layer.Layer;
import org.gitia.froog.lossfunction.LossFunction;
import org.gitia.froog.trainingalgorithm.Backpropagation;
import org.gitia.froog.transferfunction.TransferFunction;
import org.gitia.jdataanalysis.CSV;

/**
 *
 * @author Matías Roodschild <mroodschild@gmail.com>
 */
public class Funcion {

    public static void main(String[] args) {
            SimpleMatrix input = CSV.open("src/main/resources/function/train_in.csv");
            SimpleMatrix output = CSV.open("src/main/resources/function/train_out.csv");
            SimpleMatrix in_test = CSV.open("src/main/resources/function/test_in.csv");
            SimpleMatrix out_test = CSV.open("src/main/resources/function/test_out.csv");
            SimpleMatrix all_in = CSV.open("src/main/resources/function/all_in.csv");
            
            int inputSize = input.numCols();
            int outputSize = output.numCols();
            
            //==================== Preparamos la RNA =======================
            Random rand = new Random(1);
            int nn = 8;
            Feedforward net = new Feedforward();
            net.addLayer(new Layer(inputSize, nn, TransferFunction.TANSIG, rand));
            net.addLayer(new Layer(nn, outputSize, TransferFunction.PURELIM, rand));
            
            //=================  configuraciones del ensayo ========================
            // Preparamos el algoritmo de entrenamiento
            Backpropagation bp = new Backpropagation();
            bp.setEpoch(1000);
            bp.setMomentum(0.9);
            bp.setLearningRate(0.01);
            bp.setInputTest(in_test);
            bp.setOutputTest(out_test);
            bp.setTestFrecuency(1);
            bp.setLossFunction(LossFunction.RMSE);
            
            input.printDimensions();
            output.printDimensions();
            bp.entrenar(net, input, output);
            
        try{
            net.outputAll(all_in).saveToFileCSV("src/main/resources/function/res_train.csv");
        } catch (IOException ex) {
            Logger.getLogger(Funcion.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}