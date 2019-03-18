import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

import com.google.protobuf.TextFormat;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import javax.imageio.ImageIO;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.Frame;
import org.opencv.core.Mat;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;
import static  org.bytedeco.javacpp.opencv_core.*;
import static  org.bytedeco.javacpp.opencv_imgproc.putText;
import static  org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * Java inference for the Object Detection API at:
 * https://github.com/tensorflow/models/blob/master/research/object_detection/
 */
public class DetectObjects {

    private final Stack<Frame> stack = new Stack();

    static List<Tensor<?>> outputs = null;

    public void detectObjects(String models, String[] labels) throws Exception {


        try (SavedModelBundle model = SavedModelBundle.load(models, "serve")) {
            printSignature(model);
            //for (int arg = 2; arg < args.length; arg++) {
            // final String filename = args[arg];

            //Stack<Frame> frameStack = new Stack<>();


            //         for (BufferedImage img3:frameList1) {


            try (Tensor<UInt8> input = makeImageTensor(imageName)) {
                outputs =
                        model
                                .session()
                                .runner()
                                .feed("image_tensor", input)
                                .fetch("detection_scores")
                                .fetch("detection_classes")
                                .fetch("detection_boxes")
                                .run();
            }

            try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
                 Tensor<Float> classesT = outputs.get(1).expect(Float.class);
                 Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
                // All these tensors have:
                // - 1 as the first dimension
                // - maxObjects as the second dimension
                // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
                // This can be verified by looking at scoresT.shape() etc.
                int maxObjects = (int) scoresT.shape()[1];
                float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
                float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
                float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];

                // Print all objects whose score is at least 0.5.
                System.out.printf("* %s\n", imageName);
                boolean foundSomething = false;
                int noOfDetections=0;
                for (int i = 0; i < scores.length; ++i) {
                    if (scores[i] < 0.5) {
                        continue;
                    }
                    foundSomething = true;
                    noOfDetections++;
                    System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
                }
              //  drawBoundingBox(imageName);


                System.out.println("number of Detected objects:"+noOfDetections);
                if (!foundSomething) {
                    System.out.println("No objects detected with a high enough score.");
                }
            }
        }
    }

    public void drawBoundingBox(Frame frame, opencv_core.Mat matFrame) throws IOException {

        if (invalidData(frame,matFrame)) return;

        ArrayList<Tensor<?>> detectedObjects = new ArrayList<>(outputs);

        Tensor<Float> scoresT1 = detectedObjects.get(0).expect(Float.class);
        //Tensor<Float> classesT1 = outputs.get(1).expect(Float.class);
        Tensor<Float> boxesT1 = detectedObjects.get(2).expect(Float.class);


        int maxObjects1 = (int) scoresT1.shape()[1];
        float[] scores1 = scoresT1.copyTo(new float[1][maxObjects1])[0];
        //float[] classes1 = classesT.copyTo(new float[1][maxObjects])[0];
        float[][] boxes1 = boxesT1.copyTo(new float[1][maxObjects1][4])[0];




        Java2DFrameConverter conv = new Java2DFrameConverter();
       // BufferedImage orginalImage = ImageIO.read(new File(name));
        BufferedImage orginalImage = conv.convert(frame);


        BufferedImage bi = new BufferedImage(orginalImage.getWidth(),orginalImage.getHeight(),BufferedImage.TYPE_INT_RGB);

        Graphics2D g = (Graphics2D)bi.getGraphics();
        g.drawImage(orginalImage,0,0,null);
        System.out.println(bi.getType());
        int w = bi.getWidth();
        int h = bi.getHeight();
        int bufferSize = w*h*3;
           for (int bb = 0; bb < 5; bb++) {
                System.out.println("-----------------------------------");
                int ymin = Math.round(boxes1[bb][0] * h);
                int xmin = Math.round(boxes1[bb][1] * w);
                int ymax = Math.round(boxes1[bb][2] * h);
                int xmax = Math.round(boxes1[bb][3] * w);




                System.out.println("X1 " + xmin + " Y1 " + ymin + " X2 " + xmax + " Y2 " + ymax);
                // System.out.println("Score " + detection_scores[0][i]);
                //   System.out.println("Predicted " + detection_classes[0][i]);
//

                g.setColor(Color.RED);
                g.drawRect(xmin, ymin, xmax - xmin, ymax - ymin);
               // g.drawString(labels[Math.round(detection_classes[0][i])], xmin, ymin);

        }
        //ImageIO.write(bi,"PNG",new File("images/result" + System.currentTimeMillis() + ".png"));




    }
    private boolean invalidData(Frame frame, opencv_core.Mat matFrame){
        return outputs == null || matFrame==null || frame == null;
    }





    private static void printSignature(SavedModelBundle model) throws Exception {
        MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
        SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
        int numInputs = sig.getInputsCount();
        int i = 1;
        System.out.println("MODEL SIGNATURE");
        System.out.println("Inputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                    i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
        }
        int numOutputs = sig.getOutputsCount();
        i = 1;
        System.out.println("Outputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                    i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
        }
        System.out.println("-----------------------------------------------");
    }

    public static String[] loadLabels(String filename) throws Exception {
        String text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
        StringIntLabelMap.Builder builder = StringIntLabelMap.newBuilder();
        TextFormat.merge(text, builder);
        StringIntLabelMap proto = builder.build();
        int maxId = 0;
        for (StringIntLabelMapItem item : proto.getItemList()) {
            if (item.getId() > maxId) {
                maxId = item.getId();
            }
        }
        String[] ret = new String[maxId + 1];
        for (StringIntLabelMapItem item : proto.getItemList()) {
            ret[item.getId()] = item.getDisplayName();
        }
        return ret;
    }

    private static void bgr2rgb(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
        Java2DFrameConverter conv = new Java2DFrameConverter();
        //BufferedImage img = conv.convert(frame1);
        BufferedImage img = ImageIO.read(new File(filename));
       // BufferedImage img = filename;
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                    String.format(
                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                            img.getType(), "man.png"));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
    }


    public void push(Frame matFrame){
        stack.push(matFrame);
    }
    private static void printUsage(PrintStream s) {
        s.println("USAGE: <model> <label_map> <image> [<image>] [<image>]");
        s.println("");
        s.println("Where");
        s.println("<model> is the path to the SavedModel directory of the model to use.");
        s.println("        For example, the saved_model directory in tarballs from ");
        s.println(
                "        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)");
        s.println("");
        s.println(
                "<label_map> is the path to a file containing information about the labels detected by the model.");
        s.println("            For example, one of the .pbtxt files from ");
        s.println(
                "            https://github.com/tensorflow/models/tree/master/research/object_detection/data");
        s.println("");
        s.println("<image> is the path to an image file.");
        s.println("        Sample images can be found from the COCO, Kitti, or Open Images dataset.");
        s.println(
                "        See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md");
    }
}