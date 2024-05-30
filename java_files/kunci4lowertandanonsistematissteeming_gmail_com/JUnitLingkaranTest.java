package kunci4lowertandanonsistematissteeming_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitLingkaranTest {
	  private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
	  private final PrintStream oPrintStream = System.out;

	  @Before
	  public void setUpStream() {
	    System.setOut(new PrintStream(outputStream));
	  }

	  @After
	  public void restoreStream() {
	    System.setOut(oPrintStream);
	  }

	  @Test
	  public void isLingkaran() {
		double dataLuas = kunci4lowertandanonsistematissteeming_gmail_com.Lingkaran.luasLingkaran(3.14, 14);
		double expectedOutputLuas = 615.44;
	    assertEquals("calculate luas not the same",expectedOutputLuas,Double.valueOf(dataLuas), 0.01);
	    
	    double dataKeliling = kunci4lowertandanonsistematissteeming_gmail_com.Lingkaran.kelilingLingkaran(3.14, 14);
		double expectedOutputKeliling = 87.92;
	    assertEquals("calculate luas not the same",expectedOutputKeliling,Double.valueOf(dataKeliling), 0.01);
	    
	    kunci4lowertandanonsistematissteeming_gmail_com.Lingkaran.main(null);
	    String expectedOutput = "Luas Lingkaran = " + dataLuas + "\n" + "Keliling Lingkaran = " + dataKeliling; 
	    assertEquals("Output not same", expectedOutput, outputStream.toString());
	  }
}
