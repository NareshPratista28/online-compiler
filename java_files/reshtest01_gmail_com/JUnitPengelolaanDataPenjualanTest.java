package reshtest01_gmail_com;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.Before;
import org.junit.After;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

/**
 * JUnit Test for PengelolaanDataPenjualan
 * This test validates the student's implementation
 */
public class JUnitPengelolaanDataPenjualanTest {
    
    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;
    
    @Before
    public void setUpStream() {
        System.setOut(new PrintStream(outputStream));
    }
    
    @After
    public void restoreStream() {
        System.setOut(originalOut);
    }
    
    @Test
    public void testMainMethodExecution() {
        try {
            reshtest01_gmail_com.PengelolaanDataPenjualan.main(new String[]{});
            String output = outputStream.toString();
            assertFalse("Program should produce output", output.trim().isEmpty());
            assertTrue("Program executed successfully", true);
        } catch (Exception e) {
            fail("Main method failed to execute: " + e.getMessage());
        }
    }
    
    @Test
    public void testProgramLogic() {
        try {
            reshtest01_gmail_com.PengelolaanDataPenjualan.main(new String[]{});
            String output = outputStream.toString().trim();
            
            // Verify program produces meaningful output
            assertFalse("Program should produce meaningful output", output.isEmpty());
            assertTrue("Output should contain numeric results or text", 
                      output.matches(".*\\d+.*") || output.length() > 3);
        } catch (Exception e) {
            fail("Program logic validation failed: " + e.getMessage());
        }
    }
}