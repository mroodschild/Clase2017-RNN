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


import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import org.gitia.froog.Feedforward;
import org.gitia.froog.layer.Layer;
import org.gitia.froog.lossfunction.LossFunction;
import org.gitia.froog.statistics.Compite;
import org.gitia.froog.statistics.ConfusionMatrix;
import org.gitia.froog.trainingalgorithm.Backpropagation;
import org.gitia.froog.transferfunction.TransferFunction;
import org.gitia.jdataanalysis.CSV;
import org.gitia.jdataanalysis.data.stats.FilterConstantColumns;
import org.gitia.jdataanalysis.data.stats.STD;

/**
 * En este ensayo exploramos el comportamiento de los pesos de la red neuronal,
 * formamos un vector con los pesos de la capa, calculamos el tamaño del mismo,
 * y comparamos el tamaño final con el inicial para ver su "movimiento"
 *
 * @author Matías Roodschild <mroodschild@gmail.com>
 */
public class MnistExample {

    public static void main(String[] args) {

        //================== Preparación de los datos ==========================
        String dataInURL = "src/main/resources/handwrittennumbers/mnist_train_in_50000.csv";
        String dataOutURL = "src/main/resources/handwrittennumbers/mnist_train_out_50000.csv";

        String testInURL = "src/main/resources/handwrittennumbers/mnist_test_in.csv";
        String testOutURL = "src/main/resources/handwrittennumbers/mnist_test_out.csv";

        SimpleMatrix input = CSV.open(dataInURL);
        SimpleMatrix output = CSV.open(dataOutURL);
        SimpleMatrix in_test = CSV.open(testInURL);
        SimpleMatrix out_test = CSV.open(testOutURL);

        //ajustamos la desviación standard
        FilterConstantColumns filter = new FilterConstantColumns();
        filter.fit(input);//ajustamos el filtro
        System.out.println("Dimensiones iniciales");
        input.printDimensions();
        input = filter.eval(input);
        in_test = filter.eval(in_test);
        System.out.println("Dimensiones finales");
        input.printDimensions();

        STD std = new STD();
        std.fit(input);

        //convertimos los datos
        input = std.eval(input);
        in_test = std.eval(in_test);

        int inputSize = input.numCols();
        int outputSize = output.numCols();

        //================== /Preparación de los datos =========================
        //==================== Preparamos la RNA =======================
        //Preparamos el algoritmo de entrenamiento
        String funcion = TransferFunction.TANSIG;
        String funcionSalida = TransferFunction.SOFTMAX;

        //Creamos la RNA
        int NL1 = 300;
        int NL2 = 150;
        int NL100 = 100;

        int seed = 1;
        System.out.println("Seed:\t" + seed);
        Random rand = new Random(seed);

        Feedforward net = new Feedforward();
        net.addLayer(new Layer(inputSize, NL1, funcion, rand));
        net.addLayer(new Layer(NL1, NL2, funcion, rand));
        net.addLayer(new Layer(NL2, NL100, funcion, rand));
//        net.addLayer(new Layer(NL100, NL100, funcion, rand));
//        net.addLayer(new Layer(NL100, NL100, funcion, rand));
        net.addLayer(new Layer(NL100, outputSize, funcionSalida, rand));

        //=================  configuraciones del ensayo ========================
        // Preparamos el algoritmo de entrenamiento
        Backpropagation sgdv = new Backpropagation();
        sgdv.setEpoch(10);
        sgdv.setBatchSize(10);
        sgdv.setClassification(true);
        sgdv.setMomentum(0.9);
        sgdv.setLearningRate(0.01);
        sgdv.setRegularization(1e-5);
        sgdv.setInputTest(in_test);
        sgdv.setOutputTest(out_test);
        sgdv.setTestFrecuency(2000);
        sgdv.setLossFunction(LossFunction.CROSSENTROPY);
        
        sgdv.entrenar(net, input, output);
        
        SimpleMatrix out1 = Compite.eval(net.outputAll(input));

//        System.out.println("Tiempo red 1: " + time1 + " Tiempo red 2: " + time2 + " Tiempo red 3: " + time3);
        System.out.println("\nMatriz de Confusion 1 - Datos de entrenamiento");
        ConfusionMatrix confusionMatrix1 = new ConfusionMatrix();
        confusionMatrix1.eval(out1, output);
        confusionMatrix1.printStats();

        SimpleMatrix out2 = Compite.eval(net.outputAll(in_test));

        System.out.println("\nMatriz de Confusion 2 - Datos de Testeo");
        ConfusionMatrix confusionMatrix2 = new ConfusionMatrix();
        confusionMatrix2.eval(out2, out_test);
        confusionMatrix2.printStats();
    }
}