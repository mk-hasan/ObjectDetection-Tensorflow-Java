import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MobileNetRun {

    public static void main (String[]args) throws Exception {
         /*  if (args.length < 3) {
            printUsage(System.err);
            System.exit(1);
        }*/
        DetectObjects detectObj = new DetectObjects();
        final String label = "labels/mscoco_label_map.pbtxt";
        final String[] labels = detectObj.loadLabels(label);


        final String models = "models/ssd_inception_v2_coco_2017_11_17/saved_model";

        final String imageName = "images/test.jpg";
        final String fileName2 = "images/videoSample2.mp4";

        List<BufferedImage> frameList1 = new ArrayList<>();
        //frameList1 = makeFrameStack(fileName2);



        VideoProcessing vp = new VideoProcessing();
        vp.startRealTimeVideoDetection(models,labels,fileName2);
        //detectObj.detectObjects(models, labels, imageName);

    }


/*
    private static List<BufferedImage> makeFrameStack(String filename2) throws FrameGrabber.Exception, IOException {

        File videoFile = new File(filename2);

        List<BufferedImage> frameList = new ArrayList<>();
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(filename2);
        grabber.start();
        Frame frame;
        Java2DFrameConverter cnv = new Java2DFrameConverter();

        for (int ii=1;ii<grabber.getLengthInFrames();ii++){
            frame = grabber.grab();
            BufferedImage img = cnv.convert(frame);
            // ImageIO.write(img, "png", new File("images/video-frame-" + System.currentTimeMillis() + ".png"));
            frameList.add(img);
        }
        grabber.stop();

        return frameList;
    }*/

}
