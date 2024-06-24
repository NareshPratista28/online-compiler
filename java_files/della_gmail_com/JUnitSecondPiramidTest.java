package della_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitSecondPiramidTest {
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
	  public void isPiramid() {
	    della_gmail_com.SecondPiramid.main(null);
	    String expectedOutput = "*\n**\n***\n****\n*****\n";
	    assertEquals("Output not the same", expectedOutput, outputStream.toString());

	  } 

}