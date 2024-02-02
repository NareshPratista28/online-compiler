package test06042_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;


public class JUnitManipulateVariableTest {
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
	  public void isPoetry() {
		  testing.ManipulateVariable.main(null);
		  assertEquals("Result Wrong", "Pembelian Gula berjumlah 3 sehingga harga yang dibayarkan adalah 120000", outputStream.toString());

	  }
}

