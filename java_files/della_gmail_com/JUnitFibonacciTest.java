package della_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitFibonacciTest {
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
	  public void isFibonnaci() {
	    della_gmail_com.Fibonacci.main(null);
	    String expectedOutput = "Deret Fibonacci: \n0 1 1 2 3 5 8 13 ";
	    assertEquals("Output not the same", expectedOutput, outputStream.toString());

	  }
}
