package lower_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitTipeDataTest {

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
	  public void isTipeData() {
	    lower_gmail_com.TipeData.main(null);
	    String expectedOutput = "Volume balok = 420, Luas permukaan balok = 344";
	    assertEquals("Output not the same", expectedOutput, outputStream.toString());

	  }
}