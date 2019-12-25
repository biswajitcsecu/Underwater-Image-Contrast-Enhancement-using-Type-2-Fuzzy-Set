import static java.lang.System.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;
import java.lang.Math;
//import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.distribution.CauchyDistribution;


public class TypeIIFuzzyEnhance {
	public static void main(String[] args) {
		//Link system libs
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);		
		try {		
			//SummaryStatistics stats;
			CauchyDistribution x = new CauchyDistribution();
			 double[] R= new double[100000];	
			 
			//Load image	
			String filename = "/home/donox/Projects/Java/TypeIIFuzzyEnhance/src/src1.png";
			Mat srcImg = new Mat();
			srcImg.setTo(new Scalar(0));
			srcImg= Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);		
			
			// Gray Image
			Mat grayImg = new Mat();
			grayImg.setTo(new Scalar(0));
			Imgproc.cvtColor(srcImg, grayImg, Imgproc.COLOR_RGB2GRAY);		
			if(srcImg.empty()){
				System.out.println("Error file not found"); 				
				System.exit(-1);
			}
		
			//image dims
			int nrow = srcImg.height();
			int ncol = srcImg.width();
			int nch = srcImg.channels();
			System.out.println("Height: " + nrow + " Width: " + ncol + " No of Chanels: " + nch);		
			
			//convert int  to float	
			Mat gImg = new Mat();
			gImg.setTo(new Scalar(0));
			Imgproc.cvtColor(srcImg, gImg, Imgproc.COLOR_RGB2GRAY); 
			gImg.convertTo(gImg, CvType.CV_64F);

			// Step I: Fuzzification		
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) {
					double[] data = gImg.get(i, j);
					data[0] = (data[0]/ 255.0f); 
					//R[0]=x.cumulativeProbability(data[0]);
					gImg.put(i, j, data);
					//System.out.println(R);
				}
			}
			
			//Parameter measure via inear index of fuzziness		
			double sum =0;
			double a=0;
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) {
					double[] data = gImg.get(i, j);
					sum += Math.min(data[0], 1-data[0]);
				}			
			}
			a= sum/(nrow*ncol);				
		
			//Exponential fuzzy entropy
			double sumex =0;
			double b=0;
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) {
					double[] data = gImg.get(i, j);
					sumex += ((data[0])* Math.exp(1-data[0])+(1-data[0])* Math.exp(data[0])-1);
				}			
			}
			double d = (Math.exp(1)-1);
			b= sumex/(nrow*ncol*Math.sqrt(d));	
			
			//Lower membership functions 
			double alf = Math.sqrt(a+b); 
			out.print(alf);		
		
			// scale parameter		
			Mat lowImg = new Mat();	
			lowImg.setTo(new Scalar(0));
			gImg.copyTo(lowImg);
			for (int i = 0; i < nrow; i++) {
				for (int j = 0; j < ncol; j++)
				{
					double[] data = gImg.get(i, j);
					data[0] = Math.pow(data[0],1/alf);               
					lowImg.put(i, j, data);
				}
			}		
			
			//Upper membership functions 
			Mat upImg = new Mat();
			upImg.setTo(new Scalar(0));
			gImg.copyTo(upImg);
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) 
				{
					double[] data = gImg.get(i, j);
					data[0] = Math.pow(data[0],alf);               
					upImg.put(i, j, data);
				}
			}		
			
			// Hamacher-T-conorm
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) 
				{
					double[] data1 = lowImg.get(i, j);
					double[] data2 = upImg.get(i, j);
					double[] data =  gImg.get(i, j);				
					data[0] = (data1[0]+data2[0] + (alf-2)*data1[0]*data2[0])/(1-(1-alf)*data1[0]*data2[0]);               
					gImg.put(i, j, data);
				}
			}

			// Update Membership grades
			/*
			 * for (int i = 0; i < nrow; i++) { for (int j = 0; j < ncol; j++) { double[]
			 * data = gImg.get(i, j); if (data[0]<0.25) { data[0] = 2*data[0]*data[0];
			 * gImg.put(i, j, data); } else { data[0] =
			 * Math.pow((1-2*(1-data[0])*(1-data[0])),1); gImg.put(i, j, data); } } }
			 */

			//Defuzzification
			for (int i = 0; i < nrow; i++)
			{
				for (int j = 0; j < ncol; j++) {
					double[] data = gImg.get(i, j);
					data[0] = data[0]*255.f;               
					gImg.put(i, j, data);
				}
			}
		
			//Display image
			HighGui.namedWindow("SourceImage",HighGui.WINDOW_NORMAL);
			HighGui.imshow("SourceImage", grayImg);	

			//Result image and convert int datatype
			gImg.convertTo(gImg, CvType.CV_8U);
			HighGui.namedWindow("EnhancedImage",HighGui.WINDOW_NORMAL);
			HighGui.imshow("EnhancedImage", gImg);	
			Imgcodecs.imwrite("enhanced.png", gImg);
			HighGui.waitKey(0);
			HighGui.destroyAllWindows();				
		}catch(Exception e) {			
			System.out.println();
			System.err.println("Error: " + e.getMessage());	
			System.exit(-1);
		}
		System.exit(0);
	}
}


