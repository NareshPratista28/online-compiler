package lisa_gmail_com;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitOrderNestTest {
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
	    public void myOrderCaseResult() {
	    	lisa_gmail_com.OrderNest.main(null);
		    assertEquals("ShippingCost not same", "Shipping cost: 0.85", outputStream.toString());

			double data1 = lisa_gmail_com.OrderNest.calculateShipping("Express", "ship50");
		    assertEquals("Calculate Shipping not same", "0.85" ,String.valueOf(data1));
		    
		    lisa_gmail_com.OrderNest.ship(true, "Express", "ship50");
		    
	    }
}
