package lowerstopword_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitBankTest {
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
		double dataBunga = lowerstopword_gmail_com.Bank.hitungBunga(1000000, 6, 5);
		double expectedOutputBunga = 1338225.5776;
	    assertEquals("calculate not the same",expectedOutputBunga,Double.valueOf(dataBunga), 0.01);
	    
	    lowerstopword_gmail_com.Bank.main(null);
	    String expectedOutput = "Jumlah uang setelah 5 tahun: " + dataBunga; 
	    assertEquals("Output not same", expectedOutput, outputStream.toString());
	  }
}
