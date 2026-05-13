package resh12_gmail_com;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

public class JUnitHitungTotalSaldoTest {

    private ByteArrayOutputStream outContent;
    private PrintStream originalOut;

    @Before
    public void setUpOutput() {
        outContent = new ByteArrayOutputStream();
        originalOut = System.out;
        System.setOut(new PrintStream(outContent));
    }

    @After
    public void restoreOutput() {
        System.setOut(originalOut);
    }

    @Test
    public void testMainMethodExecution() {
        HitungTotalSaldo.main(new String[]{});
        Assert.assertTrue("Output tidak boleh kosong", outContent.toString().trim().length() > 0);
    }

    @Test
    public void testProgramOutput() {
        HitungTotalSaldo.main(new String[]{});
        String output = outContent.toString();
        Assert.assertTrue(output.contains("Detail Transaksi:"));
        Assert.assertTrue(output.contains("Saldo awal: Rp1000000.0"));
        Assert.assertTrue(output.contains("Jumlah deposito: 5"));
        Assert.assertTrue(output.contains("Jumlah penarikan: 2"));
        Assert.assertTrue(output.contains("Total saldo: Rp15800000.0"));
    }

    @Test
    public void testSpecificFunctionality() {
        HitungTotalSaldo.main(new String[]{});
        String output = outContent.toString();
        double totalSaldo = Double.parseDouble(output.split("Total saldo: ")[1].split("\\.")[0]);
        Assert.assertEquals(15800000, totalSaldo, 0);
    }

    @Test
    public void testEdgeCases() {
        HitungTotalSaldo.main(new String[]{});
        String output = outContent.toString();
        Assert.assertFalse(output.contains("Error:"));
        Assert.assertFalse(output.contains("Exception:"));
    }
}