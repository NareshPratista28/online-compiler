package reshtest01_gmail_com;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class JUnitHitungJarakTempuhTest {

    private final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    private final PrintStream originalPrintStream = System.out;

    @Before
    public void setUpStream() {
        System.setOut(new PrintStream(outputStream));
    }

    @After
    public void restoreStream() {
        System.setOut(originalPrintStream);
    }

    @Test
    public void testProgramExecution() {
        try {
            reshtest01_gmail_com.HitungJarakTempuh.main(null);
            
            String output = outputStream.toString().trim();
            assertTrue("Program harus menghasilkan output", !output.isEmpty());
            assertTrue("Output harus informatif", output.length() >= 3);
            
        } catch (Exception e) {
            fail("Program error: " + e.getMessage());
        }
    }
    
    @Test
    public void testMainMethodExists() {
        try {
            reshtest01_gmail_com.HitungJarakTempuh.main(new String[]{});
            assertTrue("Main method dapat dijalankan tanpa error", true);
        } catch (Exception e) {
            fail("Main method error: " + e.getMessage());
        }
    }
}