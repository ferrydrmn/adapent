package com.adepent.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.view.LayoutInflater;

public class LoadingDialog {

    private Activity activity;
    private AlertDialog alertDialog;

    LoadingDialog(Activity activity) {
        this.activity = activity;
    }

    void startAlertLoadingDialog() {

        AlertDialog.Builder builder = new AlertDialog.Builder(this.activity);

        LayoutInflater inflater = activity.getLayoutInflater();

        builder.setView(inflater.inflate(R.layout.dialog_loading, null));
        builder.setCancelable(false);

        this.alertDialog = builder.create();
        this.alertDialog.show();

    }

    void dismissAlertLoadingDialog() {

        this.alertDialog.dismiss();

    }
}
