package lowertandanonsistematissteeming_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitAverageTest {
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
	  public void isAngka() {
	    lowertandanonsistematissteeming_gmail_com.Average.main(null);
	    double expectedOutput = 83.33;
	    assertEquals("Output not the same",expectedOutput,Double.parseDouble(outputStream.toString()), 0.01);

	  }
}